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

# ---------------- 核心演算法: V18.0 Hyper-Flow & Congestion Control ----------------

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
            
            # 邊界與起點檢查
            if not (0 <= start[0] < rows and 0 <= start[1] < cols): return None, None, None, False
            if grid_data[start[0]][start[1]] == -1: return None, None, None, False
            if start == goal: return [ (start, start_time) ], start_time, start_dir, False

            # Optimization: 快速預檢查 (如果直線距離太遠且不忽略障礙，直接放棄深搜，節省算力)
            # if not ignore_others and (abs(start[0]-goal[0]) + abs(start[1]-goal[1])) > 100:
            #    return None, None, None, False
            
            # Constants
            MOVE_COST = 1.0
            TURN_COST = 1.5 # 降低轉向成本，讓路徑更靈活
            WAIT_COST = 1.0
            # 降低成本，鼓勵走空地，但不要因為一點點繞路就放棄
            EMPTY_SHELF_COST = 1.5 
            OCCUPIED_SHELF_COST = 3.0 # 稍微降低，避免完全不敢經過有人貨架
            
            if idle_obstacles is None: idle_obstacles = set()
            
            # 搜索深度控制：CheckOnly時極短，正常時適中
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

                # 預取預約資料，減少 dict lookup
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

                    # 動態障礙檢查
                    if not ignore_others:
                        if (nr, nc) in reserved_now: continue
                        if ((nr, nc), current) in edge_reserved_now: continue
                        if (nr, nc) in idle_obstacles and (nr, nc) != start and (nr, nc) != goal: continue

                    step_cost = MOVE_COST
                    
                    # 地形成本
                    is_occupied = ((nr, nc) in self.shelf_occupancy)
                    is_claimed = ((nr, nc) in self.claimed_spots)
                    
                    if is_occupied or is_claimed:
                        if (nr, nc) != goal and (nr, nc) != start:
                            step_cost += (9999.0 if is_loaded else OCCUPIED_SHELF_COST)
                    elif grid_data[nr][nc] == 1: 
                        step_cost += EMPTY_SHELF_COST
                    elif grid_data[nr][nc] == 4: # 排隊區
                        step_cost = MOVE_COST 

                    # 轉向
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
        self.capacity = capacity
        self.assignments = defaultdict(dict) 
        self.inbound_counts = defaultdict(int) # 追踪前往該站的 AGV 數量
        self.exit_points = {}
        self._init_exits()

    def _init_exits(self):
        for sid, info in self.stations.items():
            er, ec = info['pos']
            # 簡單的出口邏輯：左邊去左邊，右邊去右邊
            self.exit_points[sid] = (er, 1) if ec < 10 else (er, 6)

    def can_add_inbound(self, sid):
        # 擁塞控制核心：限制前往同一工作站的 AGV 數量
        # 如果已經有 3 台車在路上或排隊，就拒絕新訂單
        return self.inbound_counts[sid] < 3

    def register_inbound(self, sid):
        self.inbound_counts[sid] += 1

    def deregister_inbound(self, sid):
        if self.inbound_counts[sid] > 0:
            self.inbound_counts[sid] -= 1

    # 簡化版的進站邏輯，不再搞複雜的 slot 分配，依賴 A* 和 Ghost
    def request_access(self, sid, agv_id):
        return self.stations[sid]['pos']

class BatchWriter:
    def __init__(self, filepath, header):
        self.f = open(filepath, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.f)
        self.writer.writerow(header)
    def writerow(self, row): self.writer.writerow(row)
    def close(self): self.f.close()

class SimulationRunner:
    def __init__(self):
        print(f"🚀 [Core V18.0] 啟動模擬 (Hyper-Flow & Congestion Control)...")
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
        # 激進清理，只保留最近的
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
            # 減少 AGV 數量以換取流暢度 (如果不需要那麼多)
            target_count_2f = 60 # 微調
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
        # 關鍵優化：鎖定時間極短，避免「隱形牆」
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
        # 移動結束後，只鎖定極短時間 (1秒)，讓別人有機會規劃路徑
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
            if 0<=nr<rows and 0<=nc<cols and grid[nr][nc] == 1: # 找空地
                if (nr, nc) not in self.shelf_occupancy[floor]:
                    return (nr, nc)
        return start_pos

    # ---------------- 執行邏輯 ----------------
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

            # Init Events
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
                    
                    if state['time'] > sim_time + 120: continue # 別跑太快
                    if state['time'] > sim_time: sim_time = state['time']

                    curr_pos = state['pos']
                    curr_time = state['time']
                    current_obstacles = static_obstacles_cache[floor]
                    
                    # --- STATUS 1: IDLE & TASK HUNTING ---
                    if state['status'] == 'IDLE':
                        # 1. 檢查是否擋路 (簡單版)
                        is_blocking = (curr_pos in self.claimed_spots[floor])
                        if is_blocking:
                            dodge_pos = self._find_random_nearby(floor, curr_pos, 3)
                            if dodge_pos != curr_pos:
                                path, _, _, _ = astar.find_path(curr_pos, dodge_pos, curr_time, idle_obstacles=current_obstacles)
                                if path:
                                    self._execute_move(floor, agv_id, path, 'YIELD')
                                    continue

                        # 2. 接單邏輯 (加入 Congestion Control)
                        best_task = None
                        # 簡單的距離排序
                        my_stations = []
                        for sid, q in task_queues[floor].items():
                            if q:
                                dist = abs(curr_pos[0]-self.stations[sid]['pos'][0]) + abs(curr_pos[1]-self.stations[sid]['pos'][1])
                                my_stations.append((dist, sid))
                        my_stations.sort()
                        
                        for _, sid in my_stations:
                            # [關鍵優化] 如果這個站已經有太多車要去，就跳過
                            if not self.zm[floor].can_add_inbound(sid): continue
                            
                            # 檢查取貨路徑連通性 (忽略障礙，只看地形)
                            task = task_queues[floor][sid][0]
                            shelf_pos = self.shelf_coords[task['shelf_id']]['pos']
                            
                            path_check, _, _, _ = astar.find_path(curr_pos, shelf_pos, curr_time, check_only=True, ignore_others=True)
                            if path_check:
                                best_task = task_queues[floor][sid].popleft()
                                self.zm[floor].register_inbound(sid)
                                #print(f"📦 [Assign] AGV_{agv_id} -> {best_task['shelf_id']} -> {sid}")
                                break
                        
                        if best_task:
                            self.agv_tasks[agv_id] = best_task
                            state['status'] = 'MOVING_TO_PICK'
                            state['patience'] = 0
                        else:
                            # IDLE 時只鎖定 1 秒，讓其他車可以規劃穿過我的路徑(只要我下一秒不在)
                            state['time'] += 1
                            self._lock_spot(floor, curr_pos, curr_time, 1)

                    # --- STATUS 2: MOVING TO PICK ---
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
                            # Path blocked
                            state['patience'] += 1
                            if state['patience'] > 2: # 稍微等待一下就 Ghost
                                ghost_path, _, _, _ = astar.find_path(curr_pos, target_pos, curr_time, ignore_others=True)
                                if ghost_path:
                                    #print(f"👻 [Ghost-Pick] AGV_{agv_id} 穿牆取貨")
                                    self._execute_move(floor, agv_id, ghost_path, 'GHOST_MOVE', 'ForcePick')
                                    if target_pos in self.shelf_occupancy[floor]: self.shelf_occupancy[floor].remove(target_pos)
                                    state['status'] = 'LOADED'
                                    state['time'] += 5
                                else:
                                    # 極端情況：瞬移
                                    state['pos'] = target_pos
                                    state['time'] += 10
                                    state['status'] = 'LOADED'
                            else:
                                state['time'] += 1
                                self._lock_spot(floor, curr_pos, curr_time, 1)

                    # --- STATUS 3: LOADED (TO STATION) ---
                    elif state['status'] == 'LOADED':
                        task = self.agv_tasks[agv_id]
                        sid = task['stops'][0]['station']
                        st_pos = self.stations[sid]['pos']
                        
                        dist = abs(curr_pos[0]-st_pos[0]) + abs(curr_pos[1]-st_pos[1])
                        
                        # 正常路徑
                        path, end_t, _, _ = astar.find_path(curr_pos, st_pos, curr_time, idle_obstacles=current_obstacles, is_loaded=True)
                        
                        if path:
                            # 檢查是否到達工作站範圍
                            if path[-1][0] == st_pos or dist < 2:
                                self._execute_move(floor, agv_id, path, 'AGV_MOVE', f"To {sid}")
                                # 執行工作
                                proc_time = task['stops'][0]['time']
                                state['time'] += proc_time
                                self._lock_spot(floor, st_pos, state['time']-proc_time, proc_time)
                                
                                # KPI
                                finish_ts = state['time']
                                wave_id = task.get('wave_id', 'UNK')
                                self.kpi_writer.writerow([self.to_dt(finish_ts), 'INBOUND', wave_id, 'N', self.to_dt(finish_ts).date(), sid, self.wave_totals[wave_id], 0])
                                
                                state['status'] = 'RETURNING'
                                self.zm[floor].deregister_inbound(sid)
                                done_count += 1
                                if done_count % 100 == 0:
                                    print(f"📈 [進度] 已完成: {done_count}/{self.total_tasks} ({done_count/self.total_tasks*100:.1f}%) | Time: {sim_time}s")
                            else:
                                self._execute_move(floor, agv_id, path, 'AGV_MOVE')
                                state['patience'] = 0
                        else:
                            state['patience'] += 1
                            if state['patience'] > 3: # 進站卡住，快速 Ghost
                                ghost_path, _, _, _ = astar.find_path(curr_pos, st_pos, curr_time, ignore_others=True)
                                if ghost_path:
                                    #print(f"👻 [Ghost-In] AGV_{agv_id} 穿牆進站 {sid}")
                                    self._execute_move(floor, agv_id, ghost_path, 'GHOST_MOVE', 'ForceIn')
                                    # 強制結算
                                    state['time'] += 20
                                    state['status'] = 'RETURNING'
                                    self.zm[floor].deregister_inbound(sid)
                                    done_count += 1
                                else:
                                    # 瞬移結算
                                    state['pos'] = st_pos
                                    state['time'] += 20
                                    state['status'] = 'RETURNING'
                                    self.zm[floor].deregister_inbound(sid)
                                    done_count += 1
                            else:
                                state['time'] += 1
                                self._lock_spot(floor, curr_pos, curr_time, 1)

                    # --- STATUS 4: RETURNING ---
                    elif state['status'] == 'RETURNING':
                        task = self.agv_tasks[agv_id]
                        shelf_id = task['shelf_id']
                        orig_pos = self.shelf_coords[shelf_id]['pos']
                        
                        target = orig_pos
                        # 簡單的緩衝邏輯：如果原位有人，就隨便找個附近的空位
                        if target in self.shelf_occupancy[floor] or target in self.claimed_spots[floor]:
                             target = self._find_random_nearby(floor, orig_pos)
                        
                        self.claimed_spots[floor].add(target)
                        
                        path, end_t, _, _ = astar.find_path(curr_pos, target, curr_time, idle_obstacles=current_obstacles, is_loaded=True)
                        
                        if path:
                            self._execute_move(floor, agv_id, path, 'AGV_MOVE')
                            self.shelf_occupancy[floor].add(target)
                            self.shelf_coords[shelf_id]['pos'] = target # Update new pos
                            self.event_writer.writerow([self.to_dt(end_t), self.to_dt(end_t+5), floor, f"AGV_{agv_id}", target[1], target[0], target[1], target[0], 'SHELF_UNLOAD', shelf_id])
                            
                            state['status'] = 'IDLE'
                            self.claimed_spots[floor].remove(target)
                            del self.agv_tasks[agv_id]
                        else:
                            state['patience'] = state.get('patience', 0) + 1
                            if state['patience'] > 3:
                                ghost_path, _, _, _ = astar.find_path(curr_pos, target, curr_time, ignore_others=True)
                                if ghost_path:
                                    #print(f"👻 [Ghost-Return] AGV_{agv_id} 穿牆歸位")
                                    self._execute_move(floor, agv_id, ghost_path, 'GHOST_MOVE')
                                    self.shelf_occupancy[floor].add(target)
                                    self.shelf_coords[shelf_id]['pos'] = target
                                    state['status'] = 'IDLE'
                                    self.claimed_spots[floor].remove(target)
                                    del self.agv_tasks[agv_id]
                                else:
                                    # Teleport
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
            print("🎉 V18.0 模擬結束")

if __name__ == "__main__":
    SimulationRunner().run()