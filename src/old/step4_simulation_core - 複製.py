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

# ---------------- 核心演算法: V17.0 Ultimate Ghost & Strict Zoning ----------------

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
            res_get = self.reservations.get
            edge_res_get = self.edge_reservations.get
            shelf_occ = self.shelf_occupancy
            claimed = self.claimed_spots
            moves = self.moves
            
            # 邊界與起點檢查
            if not (0 <= start[0] < rows and 0 <= start[1] < cols): return None, None, None, False
            if grid_data[start[0]][start[1]] == -1: return None, None, None, False
            
            # Constants
            MOVE_COST = 1.0
            TURN_COST = 2.0
            WAIT_COST = 1.0
            EMPTY_SHELF_COST = 2.0
            OCCUPIED_SHELF_COST = 0
            LARGE_PENALTY = 9999.0
            
            if idle_obstacles is None: idle_obstacles = set()
            
            max_depth = 1500 if not check_only else 150 
            
            g_r, g_c = goal
            start_h = abs(start[0] - g_r) + abs(start[1] - g_c)
            
            open_set = []
            heapq.heappush(open_set, (0, start_h, start_time, start, start_dir))
            g_score = {(start, start_time, start_dir): 0}
            came_from = {}
            
            steps = 0
            final_node = None
            has_conflict = False 
            
            while open_set:
                steps += 1
                if steps > max_depth: break
                
                f, h, current_time, current, current_dir = heapq.heappop(open_set)
                
                if current == goal:
                    final_node = (current, current_time, current_dir)
                    break
                
                current_state_key = (current, current_time, current_dir)
                current_g = g_score.get(current_state_key, float('inf'))
                if current_g < (f - h): continue

                cr, cc = current
                next_time = current_time + 1

                reserved_now = None
                edge_reserved_now = None
                
                if not check_only and not ignore_others:
                    reserved_now = res_get(next_time)
                    edge_reserved_now = edge_res_get(current_time)

                for i, (dr, dc) in enumerate(moves):
                    nr, nc = cr + dr, cc + dc
                    next_dir = i
                    
                    # 1. 基本物理檢查
                    if not (0 <= nr < rows and 0 <= nc < self.cols): continue
                    if grid_data[nr][nc] == -1: continue 

                    # 2. 動態預約檢查 (ignore_others=True 時跳過)
                    if not check_only and not ignore_others:
                        if reserved_now and (nr, nc) in reserved_now: continue
                        if edge_reserved_now and ((nr, nc), current) in edge_reserved_now: continue

                    # 3. IDLE 車輛檢查 (ignore_others=True 時跳過)
                    if not ignore_others:
                        if (nr, nc) in idle_obstacles and (nr, nc) != start:
                            continue 

                    step_cost = MOVE_COST
                    
                    # 4. 貨架與排隊區檢查
                    is_physically_occupied = ((nr, nc) in shelf_occ)
                    is_claimed = ((nr, nc) in claimed)
                    
                    if is_physically_occupied or is_claimed:
                        if (nr, nc) == goal or (nr, nc) == start: 
                            pass 
                        else:
                            if is_loaded: 
                                step_cost += LARGE_PENALTY
                            else:
                                step_cost += OCCUPIED_SHELF_COST
                    elif grid_data[nr][nc] == 1: 
                        step_cost += EMPTY_SHELF_COST
                    elif grid_data[nr][nc] == 4:
                        step_cost = MOVE_COST 

                    # 5. 轉向成本
                    if dr == 0 and dc == 0: 
                        step_cost = WAIT_COST
                        next_dir = current_dir
                    else:
                        if current_dir != 4 and next_dir != current_dir:
                            step_cost += TURN_COST
                    
                    new_g = current_g + step_cost
                    
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
                    if pos in idle_obstacles and pos != start and pos != goal:
                        has_conflict = True
                    path.append((pos, t))
                    curr = came_from[curr]
                path.append((start, start_time))
                path.reverse()
                return path, path[-1][1], final_node[2], has_conflict
            return None, None, None, False

class MapAnalyzer:
    def __init__(self, grid):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.connectivity_map = self._build_connectivity_map()

    def _build_connectivity_map(self):
        c_map = {}
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == -1: continue 
                neighbors = 0
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] != -1:
                        neighbors += 1
                c_map[(r, c)] = neighbors
        return c_map

    def find_safe_buffer(self, start_pos, occupied_set, claimed_set):
        q = deque([start_pos])
        visited = {start_pos}
        best_candidate = None
        while q:
            curr = q.popleft()
            if self.grid[curr[0]][curr[1]] == 1: 
                if curr not in occupied_set and curr not in claimed_set:
                    degree = self.connectivity_map.get(curr, 4)
                    if degree <= 1: return curr 
                    if degree == 2 and not best_candidate: best_candidate = curr 
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = curr[0]+dr, curr[1]+dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if (nr, nc) not in visited and self.grid[nr][nc] != -1:
                        visited.add((nr, nc))
                        q.append((nr, nc))
        return best_candidate if best_candidate else start_pos

class PhysicalZoneManager:
    def __init__(self, stations_info, grid, capacity=4):
        self.stations = stations_info
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.capacity = capacity
        self.slots_map = {} 
        self.assignments = defaultdict(dict) 
        self.inbound_counts = defaultdict(int)
        self.exit_points = {}
        self.wait_queues = defaultdict(deque)
        self._init_slots()

    def is_processing(self, sid, agv_id):
        if sid in self.assignments and agv_id in self.assignments[sid]:
            return True
        return False

    def get_assigned_spot(self, sid, agv_id):
        if self.is_processing(sid, agv_id):
            slot_idx = self.assignments[sid][agv_id]
            return self.slots_map[sid][slot_idx]
        return None

    def request_access(self, sid, agv_id):
            if self.is_processing(sid, agv_id):
                return self.get_assigned_spot(sid, agv_id)

            if agv_id not in self.wait_queues[sid]:
                self.wait_queues[sid].append(agv_id)
            
            if self.wait_queues[sid][0] == agv_id:
                spot = self.assign_spot(sid, agv_id)
                if spot:
                    self.wait_queues[sid].popleft()
                    return spot
            return None

    def _init_slots(self):
        QUEUE_MARKER = 4
        for sid, info in self.stations.items():
            center_pos = info['pos']
            valid_slots = []
            found_marker_slots = []
            max_search_dist = 10 
            q = deque([center_pos])
            visited = {center_pos}
            while q:
                curr = q.popleft()
                r, c = curr
                dist = abs(r - center_pos[0]) + abs(c - center_pos[1])
                if dist > max_search_dist: continue
                if self.grid[r][c] == QUEUE_MARKER and curr != center_pos:
                    found_marker_slots.append(curr)
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in visited:
                        if self.grid[nr][nc] != -1: 
                             visited.add((nr, nc))
                             q.append((nr, nc))
            
            if found_marker_slots:
                found_marker_slots.sort(key=lambda p: abs(p[0]-center_pos[0]) + abs(p[1]-center_pos[1]))
                valid_slots = found_marker_slots[:self.capacity]
            else:
                q_backup = deque([center_pos])
                visited_backup = {center_pos}
                while q_backup and len(valid_slots) < self.capacity:
                    curr = q_backup.popleft()
                    if curr != center_pos and self.grid[curr[0]][curr[1]] != -1:
                        valid_slots.append(curr)
                    for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                        nr, nc = curr[0]+dr, curr[1]+dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in visited_backup:
                            visited_backup.add((nr, nc))
                            q_backup.append((nr, nc))
            
            self.slots_map[sid] = valid_slots
            er, ec = center_pos
            best_exit = center_pos
            if ec < 10: best_exit = (er, 1) 
            else: best_exit = (er, 6)
            self.exit_points[sid] = best_exit

    def can_add_inbound(self, sid):
        return self.inbound_counts[sid] < 4

    def register_inbound(self, sid):
        self.inbound_counts[sid] += 1

    def deregister_inbound(self, sid):
        if self.inbound_counts[sid] > 0:
            self.inbound_counts[sid] -= 1

    def assign_spot(self, sid, agv_id):
        if sid not in self.slots_map: return None
        if agv_id in self.assignments[sid]:
            return self.slots_map[sid][self.assignments[sid][agv_id]]
        used_indices = set(self.assignments[sid].values())
        for i in range(len(self.slots_map[sid])):
            if i not in used_indices:
                self.assignments[sid][agv_id] = i
                return self.slots_map[sid][i]
        return None 

    def release_spot(self, sid, agv_id):
        if sid in self.assignments and agv_id in self.assignments[sid]:
            del self.assignments[sid][agv_id]

class BatchWriter:
    def __init__(self, filepath, header):
        self.f = open(filepath, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.f)
        self.writer.writerow(header)
    def writerow(self, row): self.writer.writerow(row)
    def close(self): self.f.close()

# ---------------- 主模擬器 V17.0 ----------------

class SimulationRunner:
    def __init__(self):
        print(f"🚀 [Core V17.0] 啟動模擬 (Ultimate Ghost & Strict Zoning)...")
        self._load_data()
        self.reservations = {'2F': defaultdict(set), '3F': defaultdict(set)}
        self.edge_reservations = {'2F': defaultdict(set), '3F': defaultdict(set)}
        self.shelf_occupancy = {'2F': set(), '3F': set()}
        self.claimed_spots = {'2F': set(), '3F': set()}
        self.pos_to_sid = {'2F': {}, '3F': {}}
        self._init_shelves()
        self.agv_state = self._init_agvs()
        self.agv_tasks = {}

        self.map_analyzer = {
            '2F': MapAnalyzer(self.grid_2f),
            '3F': MapAnalyzer(self.grid_3f)
        }
        
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
        if sim_time % 50 != 0: return
        threshold = sim_time - 10 
        for floor in ['2F', '3F']:
            expired_times = [t for t in self.reservations[floor] if t < threshold]
            for t in expired_times:
                del self.reservations[floor][t]
            expired_edges = [t for t in self.edge_reservations[floor] if t < threshold]
            for t in expired_edges:
                del self.edge_reservations[floor][t]

    def _init_shelves(self):
        for sid, info in self.shelf_coords.items():
            f, p = info['floor'], info['pos']
            if f == '2F' and self.grid_2f[p[0]][p[1]] != -1: 
                self.shelf_occupancy['2F'].add(p); self.pos_to_sid['2F'][p] = sid
            elif f == '3F' and self.grid_3f[p[0]][p[1]] != -1: 
                self.shelf_occupancy['3F'].add(p); self.pos_to_sid['3F'][p] = sid

    def _init_agvs(self):
            states = {'2F': {}, '3F': {}}
            target_count_2f = 66
            target_count_3f = 66
            pool_2f = set(self.valid_spots['2F']) | self.shelf_occupancy['2F']
            pool_3f = set(self.valid_spots['3F']) | self.shelf_occupancy['3F']
            spots_2f = list(pool_2f)
            spots_3f = list(pool_3f)
            
            actual_count_2f = min(len(spots_2f), target_count_2f)
            actual_count_3f = min(len(spots_3f), target_count_3f)
            
            seed_2f = random.sample(spots_2f, actual_count_2f)
            seed_3f = random.sample(spots_3f, actual_count_3f)
            
            for i in range(actual_count_2f): 
                states['2F'][i+1] = {
                    'time': 0, 'pos': seed_2f[i], 'dir': 4, 
                    'status': 'IDLE', 'battery': 100, 
                    'force_yield': False,
                    'taboo_list': deque(maxlen=5)
                }
            
            for i in range(actual_count_3f): 
                states['3F'][i+101] = {
                    'time': 0, 'pos': seed_3f[i], 'dir': 4, 
                    'status': 'IDLE', 'battery': 100, 
                    'force_yield': False,
                    'taboo_list': deque(maxlen=5)
                }
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
        self._lock_spot(floor, last_pos, last_t, 2)

    def _find_smart_buffer_spot(self, floor, center_pos):
            occupied = self.shelf_occupancy[floor]
            claimed = self.claimed_spots[floor]
            return self.map_analyzer[floor].find_safe_buffer(center_pos, occupied, claimed)

    def _find_yield_spot(self, floor, start_pos):
        grid = self.grid_2f if floor == '2F' else self.grid_3f
        rows, cols = grid.shape
        q = deque([start_pos])
        visited = {start_pos}
        for _ in range(50): 
            if not q: break
            curr = q.popleft()
            if curr != start_pos and \
               grid[curr[0]][curr[1]] == 0 and \
               curr not in self.shelf_occupancy[floor] and \
               curr not in self.claimed_spots[floor]:
                return curr
            moves = [(0,1),(0,-1),(1,0),(-1,0)]
            random.shuffle(moves)
            for dr, dc in moves:
                nr, nc = curr[0]+dr, curr[1]+dc
                if 0<=nr<rows and 0<=nc<cols and (nr,nc) not in visited:
                    if grid[nr][nc] != -1: 
                        visited.add((nr,nc))
                        q.append((nr,nc))
        return start_pos 

    def _find_nearest_main_road(self, floor, start_pos, taboo_list=None): 
            grid = self.grid_2f if floor == '2F' else self.grid_3f
            occupied_shelves = self.shelf_occupancy[floor]
            rows, cols = grid.shape
            banned_spots = set(taboo_list) if taboo_list else set()
            q = deque([(start_pos, 0)])
            visited = {start_pos}
            max_dist = 50 
            
            while q:
                curr, dist = q.popleft()
                if dist > max_dist: break
                r, c = curr
                is_walkable = (grid[r][c] == 1 or grid[r][c] == 4)
                if is_walkable and curr not in occupied_shelves:
                    if curr != start_pos and curr not in banned_spots:
                        return curr
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if (nr, nc) not in visited and grid[nr][nc] != -1:
                            visited.add((nr, nc))
                            q.append(((nr, nc), dist + 1))
            return None

    def get_static_obstacles(self, floor, current_sim_time):
            obstacles = set()
            for aid, s in self.agv_state[floor].items():
                obstacles.add(s['pos'])
            return obstacles

    def resolve_idle_conflict(self, floor, path, idle_obstacles):
        if not path: return
        conflict_pos = None
        for p, t in path:
            if p in idle_obstacles:
                conflict_pos = p
                break
        if conflict_pos:
            target_agv = None
            for aid, s in self.agv_state[floor].items():
                if s['status'] == 'IDLE' and s['pos'] == conflict_pos:
                    target_agv = aid
                    break
            if target_agv:
                self.agv_state[floor][target_agv]['force_yield'] = True

    # --- [New Helper] 尋找料架下方躲藏點 ---
    def _find_safe_hideout(self, floor, start_pos, current_obstacles):
        """
        [需求 2] 尋找最近的「料架下方」進行躲藏。
        """
        shelf_occ = self.shelf_occupancy[floor]
        candidates = [p for p in shelf_occ if p not in current_obstacles]
        if not candidates: return None
        best_spot = min(candidates, key=lambda p: abs(p[0]-start_pos[0]) + abs(p[1]-start_pos[1]))
        if (abs(best_spot[0]-start_pos[0]) + abs(best_spot[1]-start_pos[1])) > 30: return None
        return best_spot

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

            for floor in ['2F', '3F']:
                for aid, s in self.agv_state[floor].items():
                    self.event_writer.writerow([
                        self.to_dt(0), self.to_dt(1), floor, f"AGV_{aid}",
                        s['pos'][1], s['pos'][0], s['pos'][1], s['pos'][0],
                        'INITIAL', 'InitPos'
                    ])

            active_agvs = list(self.agv_state['2F'].keys()) + list(self.agv_state['3F'].keys())
            sim_time = 0
            done_count = 0
            global_pbar = 0
            
            while True:
                self._cleanup_reservations(sim_time)
                global_pbar += 1
                if global_pbar % 20 == 0:
                    print(f"⏱ Loop {global_pbar} | Done: {done_count}")
                    rem_tasks = sum([len(q) for f in task_queues for q in task_queues[f].values()])
                    active_working = len([a for a in active_agvs if self.agv_tasks.get(a)])
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
                    
                    if state['time'] > sim_time + 300: continue
                    if state['time'] > sim_time: sim_time = state['time']

                    curr_status = state['status']
                    curr_pos = state['pos']
                    curr_time = state['time']
                    current_idle_obstacles = static_obstacles_cache[floor]
                    
                    # --- 狀態 1: IDLE ---
                    if curr_status == 'IDLE':
                        # === [需求 2: 嚴禁佔用空料架區] ===
                        is_on_empty_slot = (
                            (self.grid_2f[curr_pos[0]][curr_pos[1]] == 1 if floor == '2F' else self.grid_3f[curr_pos[0]][curr_pos[1]] == 1)
                            and (curr_pos not in self.shelf_occupancy[floor])
                        )
                        
                        if is_on_empty_slot:
                            hideout = self._find_safe_hideout(floor, curr_pos, current_idle_obstacles)
                            if hideout and hideout != curr_pos:
                                park_path, _, _, _ = astar.find_path(curr_pos, hideout, curr_time, idle_obstacles=current_idle_obstacles)
                                if park_path:
                                    print(f"🙈 [Hide] AGV_{agv_id} 從空地 {curr_pos} 躲去料架下 {hideout}")
                                    self._execute_move(floor, agv_id, park_path, 'PARKING', 'VacateSlot')
                                    continue 
                                else:
                                    # 鬼步躲藏
                                    ghost_park_path, _, _, _ = astar.find_path(curr_pos, hideout, curr_time, ignore_others=True)
                                    if ghost_park_path:
                                         print(f"👻 [Ghost Hide] AGV_{agv_id} 穿牆躲去料架下 {hideout}")
                                         self._execute_move(floor, agv_id, ghost_park_path, 'GHOST_PARK', 'ForceVacate')
                                         continue
                        
                        # [Yield Logic]
                        if state.get('force_yield'):
                            # 修改重點：優先找 "Hideout" (料架下方)，找不到才找路邊空地
                            yield_spot = self._find_safe_hideout(floor, curr_pos, current_idle_obstacles)
                            
                            # 如果附近沒料架可躲，才退而求其次找空地 (原本的邏輯)
                            if not yield_spot:
                                yield_spot = self._find_yield_spot(floor, curr_pos)
                                
                            if yield_spot and yield_spot != curr_pos:
                                path_yield, _, _, _ = astar.find_path(curr_pos, yield_spot, curr_time, check_only=False)
                                if path_yield:
                                    # 這裡加個 log 讓你知道它去哪了
                                    is_hiding = (yield_spot in self.shelf_occupancy[floor])
                                    action_name = 'SmartYield' if is_hiding else 'RoadYield'
                                    print(f"🍊 [Yield] AGV_{agv_id} 讓路 -> {yield_spot} ({action_name})")
                                    
                                    self._execute_move(floor, agv_id, path_yield, 'YIELD', action_name)
                                    state['status'] = 'IDLE' 
                                    state['force_yield'] = False
                                    continue 
                            state['force_yield'] = False

                        # [Task Hunting] - Modified V17.2 (Aggressive Assignment)
                        best_task = None
                        candidate_stations = []
                        existing_tasks = list(task_queues[floor].keys())
                        
                        # 1. 快速篩選候選工作站
                        for sid in existing_tasks:
                            if sid not in self.stations: continue
                            st_pos = self.stations[sid]['pos']
                            # 曼哈頓距離粗估
                            dist = abs(curr_pos[0] - st_pos[0]) + abs(curr_pos[1] - st_pos[1])
                            candidate_stations.append((dist, sid))

                        candidate_stations.sort(key=lambda x: x[0])
                        search_limit = 5 # 稍微放寬搜尋範圍
                        filtered_candidates = candidate_stations[:search_limit]

                        for dist, sid in filtered_candidates:
                            q = task_queues[floor][sid]
                            if not q: continue
                            if not self.zm[floor].can_add_inbound(sid): continue
                            
                            task = q[0]
                            shelf_id = task['shelf_id']
                            shelf_pos = self.shelf_coords[shelf_id]['pos']
                            
                            # --- 修改重點開始 ---
                            
                            # 步驟 A: 檢查「我去取貨點」的路徑
                            # 關鍵：使用 ignore_others=True。我們只在乎地形是否連通，不在乎現在有沒有車擋路。
                            path1, _, _, _ = astar.find_path(
                                curr_pos, shelf_pos, curr_time, 
                                idle_obstacles=None, # 不管障礙物
                                check_only=True, 
                                ignore_others=True   # 強制無視其他車
                            )
                            
                            if not path1: continue # 只有地形無法到達時才放棄
                            
                            # 步驟 B: 檢查「取貨點去工作站」的路徑
                            station_pos = self.stations[sid]['pos']
                            path2, _, _, _ = astar.find_path(
                                shelf_pos, station_pos, curr_time, # 時間只是粗估，不重要
                                idle_obstacles=None, 
                                is_loaded=True, 
                                check_only=True,
                                ignore_others=True # 強制無視其他車
                            )
                            
                            if not path2: continue

                            # 既然地形連通，直接接單！
                            # 讓 MOVING_TO_PICK 狀態去處理路上的障礙 (Yield, Rescue, Ghost)
                            best_task = q.popleft()
                            self.zm[floor].register_inbound(sid) 
                            
                            # Log 一下，確信有在接單
                            print(f"📦 [Assign] AGV_{agv_id} 接單: {shelf_id} -> {sid} (無視路況)")
                            break
                            # --- 修改重點結束 ---

                        if best_task:
                            self.agv_tasks[agv_id] = best_task
                            state['status'] = 'MOVING_TO_PICK'
                        elif state['status'] == 'IDLE': 
                            total_floor_tasks = sum([len(q) for q in task_queues[floor].values()])
                            if total_floor_tasks == 0:
                                state['time'] += 30
                                self._lock_spot(floor, curr_pos, curr_time, 30)
                            elif state['time'] <= curr_time: # 確保時間推進，避免死鎖
                                state['time'] += 5
                                self._lock_spot(floor, curr_pos, curr_time, 5)

                    # --- 狀態 2: MOVING_TO_PICK (去搬貨架) ---
                    elif curr_status == 'MOVING_TO_PICK':
                        task = self.agv_tasks[agv_id]
                        shelf_id = task['shelf_id']
                        target_pos = self.shelf_coords[shelf_id]['pos']
                        
                        path, end_t, _, conflict = astar.find_path(
                            curr_pos, target_pos, curr_time, 
                            idle_obstacles=current_idle_obstacles,
                            is_loaded=False
                        )
                        
                        if path:
                            if conflict: 
                                self.resolve_idle_conflict(floor, path, current_idle_obstacles)
                                state['time'] += 5
                                self._lock_spot(floor, curr_pos, curr_time, 5)
                                continue

                            self._execute_move(floor, agv_id, path, 'AGV_MOVE')
                            if target_pos in self.shelf_occupancy[floor]: self.shelf_occupancy[floor].remove(target_pos)
                            self.event_writer.writerow([self.to_dt(end_t), self.to_dt(end_t+5), floor, f"AGV_{agv_id}", target_pos[1], target_pos[0], target_pos[1], target_pos[0], 'SHELF_LOAD', f"{shelf_id}"])
                            state['time'] += 5
                            state['status'] = 'LOADED'
                        else:
                            # === [V17.3 核彈級鬼步: 拒絕無意義的救援] ===
                            taboo = state.get('taboo_list', deque(maxlen=5))
                            state['taboo_list'] = taboo
                            
                            # 只有在 Taboo 為空（第一次遇到困難）時，才嘗試找緩衝區
                            # 如果 Taboo 裡有東西，代表這台車已經是「累犯」，不要再救了，直接穿牆！
                            rescue_path = None
                            if len(taboo) == 0:
                                escape_spot = self._find_nearest_main_road(floor, curr_pos, taboo_list=taboo)
                                if escape_spot and escape_spot != curr_pos:
                                    rescue_path, _, _, _ = astar.find_path(curr_pos, escape_spot, curr_time, idle_obstacles=set(), is_loaded=False, ignore_others=False)

                            if rescue_path:
                                print(f"🚨 [Rescue] AGV_{agv_id} 暫時前往緩衝區 {escape_spot}...")
                                self._execute_move(floor, agv_id, rescue_path, 'RESCUE_MOVE', 'Normal')
                                state['taboo_list'].append(curr_pos)
                                state['time'] += 5 
                            else:
                                # --- 暴力區：直接無視障礙直飛目的地 ---
                                ghost_target = target_pos
                                # 注意：這裡使用了 ignore_others=True 且 idle_obstacles=set()，確保一定有路
                                ghost_path, _, _, _ = astar.find_path(curr_pos, ghost_target, curr_time, idle_obstacles=set(), is_loaded=False, ignore_others=True)
                                
                                if ghost_path:
                                    print(f"👻⚡ [NUCLEAR GHOST] AGV_{agv_id} 拒絕救援循環，直接穿牆去取貨 {ghost_target}！")
                                    self._execute_move(floor, agv_id, ghost_path, 'GHOST_MOVE', 'Nuclear')
                                    if ghost_target in self.shelf_occupancy[floor]: self.shelf_occupancy[floor].remove(ghost_target)
                                    state['status'] = 'LOADED'
                                    state['time'] += 10
                                    # 鬼步成功後，清空 Taboo，給它重新做人的機會
                                    state['taboo_list'].clear()
                                else:
                                    # 萬一連鬼步都算不出來（極少見，除非目的地在牆裡），才瞬移
                                    print(f"☠️ [FATAL] AGV_{agv_id} 空間撕裂，隨機瞬移。")
                                    # ... (保留原本的瞬移代碼)
                                    neighbors = [(0,1), (0,-1), (1,0), (-1,0)]
                                    random.shuffle(neighbors)
                                    moved = False
                                    grid = self.grid_2f if floor=='2F' else self.grid_3f
                                    for dr, dc in neighbors:
                                        nr, nc = curr_pos[0]+dr, curr_pos[1]+dc
                                        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr][nc] != -1:
                                            self.agv_state[floor][agv_id]['pos'] = (nr, nc)
                                            state['time'] += 10
                                            moved = True
                                            break
                                    if not moved: state['time'] += 5

                    # --- 狀態 3: LOADED (搬貨去工作站) ---
                    elif curr_status == 'LOADED':
                        task = self.agv_tasks[agv_id]
                        sid = task['stops'][0]['station']
                        st_center = self.stations[sid]['pos']
                        dist = abs(curr_pos[0] - st_center[0]) + abs(curr_pos[1] - st_center[1])
                        
                        target_dest = None
                        is_final_approach = False
                        force_entry = False # 新增標記
                        
                        if dist <= 3:
                            slot_pos = self.zm[floor].request_access(sid, agv_id)
                            
                            # 初始化耐心值
                            if 'patience' not in state: state['patience'] = 0
                            
                            if slot_pos:
                                target_dest = slot_pos
                                is_final_approach = True
                                state['patience'] = 0 # 成功拿到位子，重置耐心
                            else:
                                # --- 修改重點：增加耐心值判斷 ---
                                state['patience'] += 1
                                if state['patience'] > 12: # 12次迴圈 (約60秒) 還進不去
                                    print(f"😤 [Force Entry] AGV_{agv_id} 在 {sid} 門口等太久，啟動強制進站！")
                                    # 直接把目標設為工作站中心，不再管 Slot
                                    target_dest = st_center 
                                    is_final_approach = True
                                    force_entry = True 
                                    # 注意：這裡不寫 continue，讓它往下跑，去觸發 A* 或 鬼步
                                else:
                                    # 還在忍耐中，原地等待
                                    state['time'] += 5
                                    self._lock_spot(floor, curr_pos, curr_time, 5)
                                    continue # 這裡才 continue
                        else:
                            target_dest = st_center
                            is_final_approach = False
                        
                        # 下面是原本的 A* 尋路邏輯，不用動
                        # 但因為上面 force_entry 時沒 continue，
                        # 如果實體路徑不通，它就會掉到最下面的 [Ultimate Ghost Strategy]
                        # 讓紫色 AGV 穿牆進去！
                        
                        path, end_t, _, conflict = astar.find_path(
                            curr_pos, target_dest, curr_time,
                            idle_obstacles=current_idle_obstacles,
                            is_loaded=True
                        )
                        
                        if path:
                            if conflict: 
                                self.resolve_idle_conflict(floor, path, current_idle_obstacles)
                                state['time'] += 5
                                self._lock_spot(floor, curr_pos, curr_time, 5)
                                continue

                            self._execute_move(floor, agv_id, path, 'AGV_MOVE', f"To {sid}")
                            
                            if is_final_approach:
                                proc_time = task['stops'][0]['time']
                                state['time'] += proc_time
                                self._lock_spot(floor, target_dest, state['time'] - proc_time, proc_time)

                                print(f"⚙️ [Work] AGV_{agv_id} 在 {sid} 進行作業 (耗時 {proc_time}s)...")
                                
                                finish_ts = state['time']
                                wave_id = task.get('wave_id', 'UNK')
                                ttype = 'INBOUND' if 'RECEIVING' in wave_id else 'OUTBOUND'
                                deadline = self.to_dt(0) + timedelta(hours=4)
                                self.kpi_writer.writerow([self.to_dt(finish_ts), ttype, wave_id, 'N', self.to_dt(finish_ts).date(), sid, self.wave_totals[wave_id], int(deadline.timestamp())])
                                
                                state['status'] = 'RETURNING'
                                self.zm[floor].deregister_inbound(sid)
                                self.zm[floor].release_spot(sid, agv_id) 
                            else:
                                state['time'] += 2
                        else:
                            # === [V17.3 核彈級鬼步 (Loaded 版)] ===
                            taboo = state.get('taboo_list', deque(maxlen=5))
                            state['taboo_list'] = taboo
                            
                            # 同樣邏輯：有 Taboo 就別救援了，直接衝
                            rescue_path = None
                            if len(taboo) == 0:
                                escape_spot = self._find_nearest_main_road(floor, curr_pos, taboo_list=taboo)
                                if escape_spot and escape_spot != curr_pos:
                                    rescue_path, _, _, _ = astar.find_path(curr_pos, escape_spot, curr_time, idle_obstacles=set(), is_loaded=True, ignore_others=False)
                            
                            if rescue_path:
                                print(f"🚨 [Rescue] AGV_{agv_id} (LOADED) 暫時前往緩衝區 {escape_spot}...")
                                self._execute_move(floor, agv_id, rescue_path, 'RESCUE_MOVE', 'Breakout')
                                state['time'] += 10
                                state['taboo_list'].append(curr_pos) 
                            else:
                                # --- 暴力區 ---
                                ghost_target = target_dest
                                ghost_path, _, _, _ = astar.find_path(curr_pos, ghost_target, curr_time, idle_obstacles=set(), is_loaded=True, ignore_others=True)
                                
                                if ghost_path:
                                    print(f"👻⚡ [NUCLEAR GHOST] AGV_{agv_id} (LOADED) 拒絕救援循環，直接穿牆進站 {ghost_target}！")
                                    self._execute_move(floor, agv_id, ghost_path, 'GHOST_MOVE', 'Nuclear')
                                    # 這裡我們不加 Taboo，反而要清空，因為它已經成功突圍了
                                    state['taboo_list'].clear()
                                    state['time'] += 10
                                    
                                    # 強制觸發「到達」邏輯，避免它穿牆過去了卻不知道自己在工作站
                                    # 這裡做一個特殊的處理：如果它真的到了工作站範圍，我們假設它下一輪會被判定為到達
                                else:
                                    print(f"☠️ [FATAL] AGV_{agv_id} 鬼步失效 (Loaded)，隨機瞬移。")
                                    neighbors = [(0,1), (0,-1), (1,0), (-1,0)]
                                    random.shuffle(neighbors)
                                    moved = False
                                    grid = self.grid_2f if floor=='2F' else self.grid_3f
                                    for dr, dc in neighbors:
                                        nr, nc = curr_pos[0]+dr, curr_pos[1]+dc
                                        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr][nc] != -1:
                                            self.agv_state[floor][agv_id]['pos'] = (nr, nc)
                                            state['time'] += 10
                                            moved = True
                                            break
                                    if not moved:
                                        state['time'] += 5

                    # --- 狀態 4: RETURNING (嚴格物理模式 + Ghost Fallback) ---
                    elif curr_status == 'RETURNING':
                        task = self.agv_tasks[agv_id]
                        shelf_id = task['shelf_id']
                        orig_pos = self.shelf_coords[shelf_id]['pos']
                        sid = task['stops'][0]['station']
                        
                        exit_pt = self.zm[floor].exit_points.get(sid)
                        
                        target_drop = orig_pos
                        if target_drop in self.shelf_occupancy[floor]: 
                            target_drop = self._find_smart_buffer_spot(floor, orig_pos)
                        
                        current_target = target_drop
                        
                        dist_to_exit = 999
                        if exit_pt:
                            dist_to_exit = abs(curr_pos[0] - exit_pt[0]) + abs(curr_pos[1] - exit_pt[1])
                        if exit_pt and dist_to_exit > 2 and dist_to_exit < 20: 
                            st_dist = abs(curr_pos[0] - self.stations[sid]['pos'][0]) + abs(curr_pos[1] - self.stations[sid]['pos'][1])
                            if st_dist < 8:
                                current_target = exit_pt
                        
                        self.claimed_spots[floor].add(current_target)
                        
                        # [Eviction] 驅趕佔位者
                        if current_target in current_idle_obstacles:
                            blocker_id = None
                            for aid, s in self.agv_state[floor].items():
                                if s['pos'] == current_target and s['status'] == 'IDLE':
                                    blocker_id = aid
                                    break
                            if blocker_id:
                                print(f"😤 [Evict] AGV_{agv_id} (Returning) 趕走佔據空位的 AGV_{blocker_id}")
                                self.agv_state[floor][blocker_id]['force_yield'] = True
                                self.claimed_spots[floor].remove(current_target)
                                state['time'] += 5
                                self._lock_spot(floor, curr_pos, curr_time, 5)
                                continue

                        path, end_t, _, conflict = astar.find_path(
                            curr_pos, current_target, curr_time, 
                            idle_obstacles=current_idle_obstacles,
                            is_loaded=True,
                            ignore_others=False 
                        )
                        
                        if path:
                            if conflict: 
                                self.resolve_idle_conflict(floor, path, current_idle_obstacles)
                                self.claimed_spots[floor].remove(current_target) 
                                state['time'] += 5
                                self._lock_spot(floor, curr_pos, curr_time, 5)
                                continue

                            self._execute_move(floor, agv_id, path, 'AGV_MOVE')
                            
                            if current_target == exit_pt:
                                self.claimed_spots[floor].remove(current_target)
                                state['time'] += 2
                                continue
                            
                            self.shelf_occupancy[floor].add(target_drop)
                            self.claimed_spots[floor].remove(target_drop) 
                            self.shelf_coords[shelf_id]['pos'] = target_drop
                            self.pos_to_sid[floor][target_drop] = shelf_id
                            
                            self.event_writer.writerow([
                                self.to_dt(end_t), self.to_dt(end_t+5), floor, f"AGV_{agv_id}", 
                                target_drop[1], target_drop[0], target_drop[1], target_drop[0], 
                                'SHELF_UNLOAD', f"{shelf_id}"
                            ])
                            state['time'] += 5
                            state['status'] = 'IDLE'
                            del self.agv_tasks[agv_id]
                            done_count += 1
                        else:
                            # Returning 失敗 -> 嘗試 Ghost 回去
                            ghost_path, _, _, _ = astar.find_path(curr_pos, current_target, curr_time, idle_obstacles=set(), is_loaded=True, ignore_others=True)
                            if ghost_path:
                                print(f"👻⚡ [ULTIMATE GHOST] AGV_{agv_id} (Returning) 無視障礙，強行歸位 {current_target}！")
                                self._execute_move(floor, agv_id, ghost_path, 'GHOST_MOVE', 'ForceReturn')
                                self.shelf_occupancy[floor].add(target_drop)
                                self.claimed_spots[floor].remove(target_drop) 
                                self.shelf_coords[shelf_id]['pos'] = target_drop
                                self.pos_to_sid[floor][target_drop] = shelf_id
                                state['time'] += 10
                                state['status'] = 'IDLE'
                                del self.agv_tasks[agv_id]
                                done_count += 1
                            else:
                                self.claimed_spots[floor].remove(current_target)
                                state['time'] += 5
                                self._lock_spot(floor, curr_pos, curr_time, 5)

            self.event_writer.close()
            self.kpi_writer.close()
            print("🎉 V17.0 模擬結束 (Ultimate Ghost & Strict Zoning)")

if __name__ == "__main__":
    SimulationRunner().run()