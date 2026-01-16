import pandas as pd
import numpy as np
import os
import time
import heapq
import csv
import random
import pickle
from collections import defaultdict, deque, Counter
from datetime import datetime, timedelta

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
INPUT_FILE = os.path.join(BASE_DIR, 'processed_sim_data.pkl')
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------- 核心演算法: V20.0 Logical Lock & Sequential Consistency ----------------

class TimeAwareAStar:
    def __init__(self, grid, reservations, edge_reservations, shelf_occupancy_set, claimed_spots, floor_name, station_spots):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.reservations = reservations
        self.edge_reservations = edge_reservations
        self.shelf_occupancy = shelf_occupancy_set
        self.claimed_spots = claimed_spots 
        self.floor = floor_name
        self.station_spots = station_spots
        self.moves = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]

    def find_path(self, start, goal, start_time, idle_obstacles=None, start_dir=4, is_loaded=False, check_only=False, ignore_others=False):
            rows, cols = self.rows, self.cols
            grid_data = self.grid
            
            if not (0 <= start[0] < rows and 0 <= start[1] < cols): return None, None, None, False
            if grid_data[start[0]][start[1]] == -1: return None, None, None, False
            if start == goal: return [ (start, start_time) ], start_time, start_dir, False

            MOVE_COST = 1.0
            TURN_COST = 1.5 
            WAIT_COST = 1.0
            EMPTY_SHELF_COST = 1.5 
            OCCUPIED_SHELF_COST = 3.0 
            
            if idle_obstacles is None: idle_obstacles = set()
            max_depth = 1000 if not check_only else 200
            
            g_r, g_c = goal
            open_set = []
            heapq.heappush(open_set, (0, 0, start_time, start, start_dir))
            g_score = {(start, start_time, start_dir): 0}
            came_from = {}
            
            steps = 0
            final_node = None
            has_conflict = False 
            
            res_floor = self.reservations
            edge_res_floor = self.edge_reservations
            
            while open_set:
                steps += 1
                if steps > max_depth: break
                
                f, h, current_time, current, current_dir = heapq.heappop(open_set)
                
                if current == goal:
                    final_node = (current, current_time, current_dir)
                    break
                
                current_state_key = (current, current_time, current_dir)
                if g_score.get(current_state_key, float('inf')) < (f - h): continue

                cr, cc = current
                next_time = current_time + 1

                if not check_only and not ignore_others:
                    reserved_now = res_floor.get(next_time, set())
                    edge_reserved_now = edge_res_floor.get(current_time, set())
                else:
                    reserved_now = set()
                    edge_reserved_now = set()

                for i, (dr, dc) in enumerate(self.moves):
                    nr, nc = cr + dr, cc + dc
                    next_dir = i
                    
                    if not (0 <= nr < rows and 0 <= nc < cols): continue
                    if grid_data[nr][nc] == -1: continue 

                    if not ignore_others:
                        if (nr, nc) in reserved_now: continue
                        if ((nr, nc), current) in edge_reserved_now: continue
                        if (nr, nc) in idle_obstacles and (nr, nc) != start and (nr, nc) != goal: continue

                    step_cost = MOVE_COST
                    is_occupied = ((nr, nc) in self.shelf_occupancy)
                    is_claimed = ((nr, nc) in self.claimed_spots)
                    
                    if is_occupied or is_claimed:
                        if (nr, nc) != goal and (nr, nc) != start:
                            step_cost += (9999.0 if is_loaded else OCCUPIED_SHELF_COST)
                    elif grid_data[nr][nc] == 1: 
                        step_cost += EMPTY_SHELF_COST
                    elif grid_data[nr][nc] == 4: 
                        step_cost = MOVE_COST 

                    if dr == 0 and dc == 0: 
                        step_cost = WAIT_COST
                        next_dir = current_dir
                    elif current_dir != 4 and next_dir != current_dir:
                        step_cost += TURN_COST
                    
                    new_g = g_score[current_state_key] + step_cost
                    state_key = ((nr, nc), next_time, next_dir)
                    
                    if new_g < g_score.get(state_key, float('inf')):
                        g_score[state_key] = new_g
                        new_h = abs(nr - g_r) + abs(nc - g_c)
                        heapq.heappush(open_set, (new_g + new_h, new_h, next_time, (nr, nc), next_dir))
                        came_from[state_key] = current_state_key

            if final_node:
                path = []
                curr = final_node
                while curr in came_from:
                    pos, t, d = curr
                    if pos in idle_obstacles and pos != start and pos != goal: has_conflict = True
                    path.append((pos, t))
                    curr = came_from[curr]
                path.append((start, start_time))
                path.reverse()
                return path, path[-1][1], final_node[2], has_conflict
            return None, None, None, False

class PhysicalZoneManager:
    def __init__(self, stations_info, grid, capacity=4):
        self.stations = stations_info
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.inbound_counts = defaultdict(int) 
        self.exit_points = {}
        # === [V20 新增] 邏輯時間鎖 ===
        # 記錄每個工作站「最早什麼時候有空」
        self.next_free_time = defaultdict(int)
        self._init_exits()

    def _init_exits(self):
        for sid, info in self.stations.items():
            er, ec = info['pos']
            self.exit_points[sid] = (er, 1) if ec < 10 else (er, 6)

    def can_add_inbound(self, sid):
        return self.inbound_counts[sid] < 3

    def register_inbound(self, sid):
        self.inbound_counts[sid] += 1

    def deregister_inbound(self, sid):
        if self.inbound_counts[sid] > 0:
            self.inbound_counts[sid] -= 1
            
    # === [V20 核心方法] 預約邏輯時段 ===
    def book_logical_slot(self, sid, arrival_time, duration):
        """
        確保任務在時間軸上不重疊。
        如果到達時間早於工作站空閒時間，就必須排隊（邏輯上）。
        回傳：實際完成時間
        """
        start_time = max(arrival_time, self.next_free_time[sid])
        finish_time = start_time + duration
        
        # 更新該站的下次空閒時間
        self.next_free_time[sid] = finish_time
        
        return finish_time

class BatchWriter:
    def __init__(self, filepath, header):
        if os.path.exists(filepath):
            try:
                with open(filepath, 'a') as test: pass
            except PermissionError:
                print(f"❌❌❌ 嚴重錯誤: 檔案 {filepath} 被鎖定！請關閉 Excel！")
                raise SystemExit("File Locked")
                
        self.f = open(filepath, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.f)
        self.writer.writerow(header)
        self.f.flush()
        os.fsync(self.f.fileno())
        self.counter = 0

    def writerow(self, row):
        try:
            self.writer.writerow(row)
            self.counter += 1
            if self.counter % 200 == 0:
                self.f.flush()
                os.fsync(self.f.fileno()) 
        except Exception as e:
            print(f"❌❌❌ 寫入失敗: {e} | 資料可能遺失！")

    def close(self):
        try:
            self.f.flush()
            os.fsync(self.f.fileno())
            self.f.close()
        except: pass

class SimulationRunner:
    def __init__(self):
        print(f"🚀 [Core V20.0] 啟動模擬 (Logical Lock & Sequential Tasks)...")
        self._load_data()
        self.reservations = {'2F': defaultdict(set), '3F': defaultdict(set)}
        self.edge_reservations = {'2F': defaultdict(set), '3F': defaultdict(set)}
        self.shelf_occupancy = {'2F': set(), '3F': set()}
        self.claimed_spots = {'2F': set(), '3F': set()}
        self._init_shelves()
        self.agv_state = self._init_agvs()
        self.agv_tasks = {}
        
        self.zm = {
            '2F': PhysicalZoneManager({k:v for k,v in self.stations.items() if v['floor']=='2F'}, self.grid_2f),  
            '3F': PhysicalZoneManager({k:v for k,v in self.stations.items() if v['floor']=='3F'}, self.grid_3f)
        }
        
        self.event_writer = BatchWriter(
            os.path.join(LOG_DIR, 'simulation_events.csv'), 
            ['start_time', 'end_time', 'floor', 'obj_id', 'sx', 'sy', 'ex', 'ey', 'type', 'text']
        )
        self.kpi_writer = BatchWriter(
            os.path.join(LOG_DIR, 'simulation_kpi.csv'), 
            ['finish_time', 'type', 'wave_id', 'is_delayed', 'date', 'workstation', 'total_in_wave', 'deadline_ts']
        )
        self.wave_totals = Counter()
        for floor in ['2F', '3F']:
            for t in self.queues[floor]:
                wid = t.get('wave_id', 'UNK')
                self.wave_totals[wid] += 1
        
        self.total_tasks = sum(len(q) for q in self.queues.values())
        print(f"📋 任務總量: {self.total_tasks} 張訂單")

    def _load_data(self):
        with open(INPUT_FILE, 'rb') as f: data = pickle.load(f)
        self.grid_2f = data['grid_2f']; self.grid_3f = data['grid_3f']
        self.stations = data['stations']; self.shelf_coords = data['shelf_coords']
        self.queues = {'2F': deque(data['queues']['2F']), '3F': deque(data['queues']['3F'])}
        self.base_time = data['base_time']
        self.valid_spots = {'2F': [], '3F': []}
        for r in range(32):
            for c in range(61):
                if self.grid_2f[r][c] == 1: self.valid_spots['2F'].append((r,c))
                if self.grid_3f[r][c] == 1: self.valid_spots['3F'].append((r,c))

    def _cleanup_reservations(self, sim_time):
        if sim_time % 20 != 0: return
        threshold = sim_time
        for floor in ['2F', '3F']:
            expired = [t for t in self.reservations[floor] if t < threshold]
            for t in expired: del self.reservations[floor][t]
            expired_edges = [t for t in self.edge_reservations[floor] if t < threshold]
            for t in expired_edges: del self.edge_reservations[floor][t]

    def _init_shelves(self):
        for sid, info in self.shelf_coords.items():
            f, p = info['floor'], info['pos']
            if f == '2F' and self.grid_2f[p[0]][p[1]] != -1: self.shelf_occupancy['2F'].add(p)
            elif f == '3F' and self.grid_3f[p[0]][p[1]] != -1: self.shelf_occupancy['3F'].add(p)

    def _init_agvs(self):
            states = {'2F': {}, '3F': {}}
            target_count_2f = 60 
            target_count_3f = 60
            pool_2f = list(set(self.valid_spots['2F']) | self.shelf_occupancy['2F'])
            pool_3f = list(set(self.valid_spots['3F']) | self.shelf_occupancy['3F'])
            random.shuffle(pool_2f); random.shuffle(pool_3f)
            
            for i in range(min(len(pool_2f), target_count_2f)): 
                states['2F'][i+1] = {'time': 0, 'pos': pool_2f[i], 'dir': 4, 'status': 'IDLE', 'patience': 0}
            for i in range(min(len(pool_3f), target_count_3f)): 
                states['3F'][i+101] = {'time': 0, 'pos': pool_3f[i], 'dir': 4, 'status': 'IDLE', 'patience': 0}
            return states

    def to_dt(self, sec): return self.base_time + timedelta(seconds=sec)

    def _lock_spot(self, floor, pos, start_t, duration):
        end_t = start_t + duration
        for t in range(int(start_t), int(end_t) + 1):
            self.reservations[floor][t].add(pos)

    def _execute_move(self, floor, agv_id, path, type_desc, info_text=""):
        if not path: return
        res_floor = self.reservations[floor]
        edge_res_floor = self.edge_reservations[floor]
        
        for i in range(len(path)-1):
            c_pos, c_t = path[i]; n_pos, n_t = path[i+1]
            res_floor[n_t].add(n_pos)
            edge_res_floor[c_t].add((c_pos, n_pos))
            self.event_writer.writerow([
                self.to_dt(c_t), self.to_dt(n_t), floor, f"AGV_{agv_id}", 
                c_pos[1], c_pos[0], n_pos[1], n_pos[0], type_desc, info_text
            ])
        
        last_pos, last_t = path[-1]
        self.agv_state[floor][agv_id]['pos'] = last_pos
        self.agv_state[floor][agv_id]['time'] = last_t
        self._lock_spot(floor, last_pos, last_t, 1)

    def get_static_obstacles(self, floor, current_sim_time):
            obstacles = set()
            for aid, s in self.agv_state[floor].items():
                obstacles.add(s['pos'])
            return obstacles

    def _find_random_nearby(self, floor, start_pos, dist=5):
        grid = self.grid_2f if floor == '2F' else self.grid_3f
        rows, cols = grid.shape
        for _ in range(10):
            dr = random.randint(-dist, dist)
            dc = random.randint(-dist, dist)
            nr, nc = start_pos[0]+dr, start_pos[1]+dc
            if 0<=nr<rows and 0<=nc<cols and grid[nr][nc] == 1:
                if (nr, nc) not in self.shelf_occupancy[floor]:
                    return (nr, nc)
        return start_pos

    def run(self):
            station_spots_2f = {info['pos'] for info in self.stations.values() if info['floor'] == '2F'}
            station_spots_3f = {info['pos'] for info in self.stations.values() if info['floor'] == '3F'}
            
            astars = {
                '2F': TimeAwareAStar(self.grid_2f, self.reservations['2F'], self.edge_reservations['2F'], self.shelf_occupancy['2F'], self.claimed_spots['2F'], '2F', station_spots_2f),
                '3F': TimeAwareAStar(self.grid_3f, self.reservations['3F'], self.edge_reservations['3F'], self.shelf_occupancy['3F'], self.claimed_spots['3F'], '3F', station_spots_3f)
            }
            
            task_queues = {'2F': defaultdict(deque), '3F': defaultdict(deque)}
            for f in ['2F', '3F']:
                while self.queues[f]:
                    t = self.queues[f].popleft()
                    sid = t['stops'][0]['station']
                    task_queues[f][sid].append(t)

            for floor in ['2F', '3F']:
                for aid, s in self.agv_state[floor].items():
                    self.event_writer.writerow([self.to_dt(0), self.to_dt(1), floor, f"AGV_{aid}", s['pos'][1], s['pos'][0], s['pos'][1], s['pos'][0], 'INITIAL', ''])

            active_agvs = list(self.agv_state['2F'].keys()) + list(self.agv_state['3F'].keys())
            sim_time = 0
            done_count = 0
            global_pbar = 0
            
            while True:
                self._cleanup_reservations(sim_time)
                global_pbar += 1
                
                # Timeout Protection (50000 loops)
                if global_pbar > 50000: 
                    if sum([len(q) for f in task_queues for q in task_queues[f].values()]) > 0:
                        print("⚠️ 達到運算上限 (50000 loops)，強制停止。")
                        break
                
                active_working = len([a for a in active_agvs if self.agv_tasks.get(a)])
                rem_tasks = sum([len(q) for f in task_queues for q in task_queues[f].values()])
                
                if rem_tasks == 0 and active_working == 0:
                    break

                static_obstacles_cache = {
                    '2F': self.get_static_obstacles('2F', sim_time),
                    '3F': self.get_static_obstacles('3F', sim_time)
                }

                all_agvs_sorted = sorted(active_agvs, key=lambda aid: self.agv_state['2F' if aid < 100 else '3F'][aid]['time'])
                
                for agv_id in all_agvs_sorted:
                    floor = '2F' if agv_id < 100 else '3F'
                    state = self.agv_state[floor][agv_id]
                    astar = astars[floor]
                    
                    if state['time'] > sim_time + 120: continue 
                    if state['time'] > sim_time: sim_time = state['time']

                    curr_pos = state['pos']
                    curr_time = state['time']
                    current_obstacles = static_obstacles_cache[floor]
                    
                    # --- STATUS 1: IDLE ---
                    if state['status'] == 'IDLE':
                        is_blocking = (curr_pos in self.claimed_spots[floor])
                        if is_blocking:
                            dodge_pos = self._find_random_nearby(floor, curr_pos, 3)
                            if dodge_pos != curr_pos:
                                path, _, _, _ = astar.find_path(curr_pos, dodge_pos, curr_time, idle_obstacles=current_obstacles)
                                if path:
                                    self._execute_move(floor, agv_id, path, 'YIELD')
                                    continue

                        best_task = None
                        my_stations = []
                        for sid, q in task_queues[floor].items():
                            if q:
                                dist = abs(curr_pos[0]-self.stations[sid]['pos'][0]) + abs(curr_pos[1]-self.stations[sid]['pos'][1])
                                my_stations.append((dist, sid))
                        my_stations.sort()
                        
                        for _, sid in my_stations:
                            if not self.zm[floor].can_add_inbound(sid): continue
                            task = task_queues[floor][sid][0]
                            shelf_pos = self.shelf_coords[task['shelf_id']]['pos']
                            path_check, _, _, _ = astar.find_path(curr_pos, shelf_pos, curr_time, check_only=True, ignore_others=True)
                            if path_check:
                                best_task = task_queues[floor][sid].popleft()
                                self.zm[floor].register_inbound(sid)
                                break
                        
                        if best_task:
                            self.agv_tasks[agv_id] = best_task
                            state['status'] = 'MOVING_TO_PICK'
                            state['patience'] = 0
                        else:
                            state['time'] += 1
                            self._lock_spot(floor, curr_pos, curr_time, 1)

                    # --- STATUS 2: MOVING_TO_PICK ---
                    elif state['status'] == 'MOVING_TO_PICK':
                        task = self.agv_tasks[agv_id]
                        target_pos = self.shelf_coords[task['shelf_id']]['pos']
                        
                        path, end_t, _, _ = astar.find_path(curr_pos, target_pos, curr_time, idle_obstacles=current_obstacles)
                        
                        if path:
                            self._execute_move(floor, agv_id, path, 'AGV_MOVE')
                            if target_pos in self.shelf_occupancy[floor]: self.shelf_occupancy[floor].remove(target_pos)
                            self.event_writer.writerow([self.to_dt(end_t), self.to_dt(end_t+5), floor, f"AGV_{agv_id}", target_pos[1], target_pos[0], target_pos[1], target_pos[0], 'SHELF_LOAD', task['shelf_id']])
                            state['time'] += 5
                            state['status'] = 'LOADED'
                            state['patience'] = 0
                        else:
                            state['patience'] += 1
                            if state['patience'] > 2: 
                                ghost_path, _, _, _ = astar.find_path(curr_pos, target_pos, curr_time, ignore_others=True)
                                if ghost_path:
                                    self._execute_move(floor, agv_id, ghost_path, 'GHOST_MOVE', 'ForcePick')
                                    if target_pos in self.shelf_occupancy[floor]: self.shelf_occupancy[floor].remove(target_pos)
                                    state['status'] = 'LOADED'
                                    state['time'] += 5
                                else:
                                    state['pos'] = target_pos
                                    state['time'] += 10
                                    state['status'] = 'LOADED'
                            else:
                                state['time'] += 1
                                self._lock_spot(floor, curr_pos, curr_time, 1)

                    # --- STATUS 3: LOADED ---
                    elif state['status'] == 'LOADED':
                        task = self.agv_tasks[agv_id]
                        sid = task['stops'][0]['station']
                        st_pos = self.stations[sid]['pos']
                        dist = abs(curr_pos[0]-st_pos[0]) + abs(curr_pos[1]-st_pos[1])
                        
                        path, end_t, _, _ = astar.find_path(curr_pos, st_pos, curr_time, idle_obstacles=current_obstacles, is_loaded=True)
                        
                        finish_logic = False
                        if path and (path[-1][0] == st_pos or dist < 2):
                             self._execute_move(floor, agv_id, path, 'AGV_MOVE', f"To {sid}")
                             finish_logic = True
                        
                        if finish_logic:
                            proc_time = task['stops'][0]['time']
                            
                            # === [V20 關鍵修改] 使用邏輯排隊鎖 ===
                            # 這會保證 finish_ts 絕對不會跟上一台車重疊
                            finish_ts = self.zm[floor].book_logical_slot(sid, state['time'], proc_time)
                            state['time'] = finish_ts # 更新 AGV 的時間，讓它"瞬移"到未來
                            
                            self._lock_spot(floor, st_pos, state['time']-proc_time, proc_time)
                            
                            wave_id = task.get('wave_id', 'UNK')
                            task_type = 'INBOUND' if 'RECEIVING' in str(wave_id) else 'OUTBOUND'
                            self.kpi_writer.writerow([self.to_dt(finish_ts), task_type, wave_id, 'N', self.to_dt(finish_ts).date(), sid, self.wave_totals[wave_id], 0])
                            
                            state['status'] = 'RETURNING'
                            self.zm[floor].deregister_inbound(sid)
                            
                            done_count += 1
                            if done_count % 100 == 0:
                                print(f"📈 [進度] 已完成: {done_count}/{self.total_tasks} ({done_count/self.total_tasks*100:.1f}%) | Time: {sim_time}s")

                        elif path:
                            self._execute_move(floor, agv_id, path, 'AGV_MOVE')
                            state['patience'] = 0
                        else:
                            state['patience'] += 1
                            if state['patience'] > 3: 
                                ghost_path, _, _, _ = astar.find_path(curr_pos, st_pos, curr_time, ignore_others=True)
                                if ghost_path:
                                    self._execute_move(floor, agv_id, ghost_path, 'GHOST_MOVE', 'ForceIn')
                                    # Ghost 也要排隊！
                                    proc_time = task['stops'][0]['time']
                                    finish_ts = self.zm[floor].book_logical_slot(sid, state['time'], proc_time)
                                    state['time'] = finish_ts
                                else:
                                    # Teleport 也要排隊！
                                    state['pos'] = st_pos
                                    proc_time = task['stops'][0]['time']
                                    finish_ts = self.zm[floor].book_logical_slot(sid, state['time'], proc_time)
                                    state['time'] = finish_ts
                                
                                # 寫入 KPI (Ghost/Teleport)
                                wave_id = task.get('wave_id', 'UNK')
                                task_type = 'INBOUND' if 'RECEIVING' in str(wave_id) else 'OUTBOUND'
                                self.kpi_writer.writerow([self.to_dt(state['time']), task_type, wave_id, 'N', self.to_dt(state['time']).date(), sid, self.wave_totals[wave_id], 0])

                                state['status'] = 'RETURNING'
                                self.zm[floor].deregister_inbound(sid)
                                done_count += 1
                                if done_count % 100 == 0:
                                    print(f"📈 [進度] 已完成: {done_count}/{self.total_tasks} ({done_count/self.total_tasks*100:.1f}%) | Time: {sim_time}s")
                            else:
                                state['time'] += 1
                                self._lock_spot(floor, curr_pos, curr_time, 1)

                    # --- STATUS 4: RETURNING ---
                    elif state['status'] == 'RETURNING':
                        task = self.agv_tasks[agv_id]
                        shelf_id = task['shelf_id']
                        orig_pos = self.shelf_coords[shelf_id]['pos']
                        
                        target = orig_pos
                        if target in self.shelf_occupancy[floor] or target in self.claimed_spots[floor]:
                             target = self._find_random_nearby(floor, orig_pos)
                        
                        self.claimed_spots[floor].add(target)
                        path, end_t, _, _ = astar.find_path(curr_pos, target, curr_time, idle_obstacles=current_obstacles, is_loaded=True)
                        
                        if path:
                            self._execute_move(floor, agv_id, path, 'AGV_MOVE')
                            self.shelf_occupancy[floor].add(target)
                            self.shelf_coords[shelf_id]['pos'] = target 
                            self.event_writer.writerow([self.to_dt(end_t), self.to_dt(end_t+5), floor, f"AGV_{agv_id}", target[1], target[0], target[1], target[0], 'SHELF_UNLOAD', shelf_id])
                            
                            state['status'] = 'IDLE'
                            self.claimed_spots[floor].remove(target)
                            del self.agv_tasks[agv_id]
                        else:
                            state['patience'] = state.get('patience', 0) + 1
                            if state['patience'] > 3:
                                ghost_path, _, _, _ = astar.find_path(curr_pos, target, curr_time, ignore_others=True)
                                if ghost_path:
                                    self._execute_move(floor, agv_id, ghost_path, 'GHOST_MOVE')
                                else:
                                    state['pos'] = target
                                    state['time'] += 10
                                self.shelf_occupancy[floor].add(target)
                                self.shelf_coords[shelf_id]['pos'] = target
                                state['status'] = 'IDLE'
                                self.claimed_spots[floor].remove(target)
                                del self.agv_tasks[agv_id]
                            else:
                                state['time'] += 1
                                self._lock_spot(floor, curr_pos, curr_time, 1)
                                self.claimed_spots[floor].remove(target)

            self.event_writer.close()
            self.kpi_writer.close()
            print("🎉 V20.0 模擬結束")

if __name__ == "__main__":
    SimulationRunner().run()