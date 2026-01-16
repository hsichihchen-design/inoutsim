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

# ---------------- 核心演算法: V16.0 Optimized ----------------

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
            # [Performance] Localize variables
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
            WAIT_COST = 10.0 
            EMPTY_SHELF_COST = 2.0
            OCCUPIED_SHELF_COST = 999.0  
            # IDLE_AGV_PENALTY 已不再需要，因為直接視為牆壁
            LARGE_PENALTY = 9999.0
            
            if idle_obstacles is None: idle_obstacles = set()
            
            max_depth = 10000 if not check_only else 5000 
            
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

                    # 2. 動態預約檢查
                    if not check_only and not ignore_others:
                        if reserved_now and (nr, nc) in reserved_now: continue
                        if edge_reserved_now and ((nr, nc), current) in edge_reserved_now: continue

                    # === [關鍵修改] 3. IDLE 車輛 = 絕對牆壁 (Strict Physics) ===
                    if not ignore_others:
                        # 如果該點有 IDLE 車，且不是起點或終點，直接跳過 (continue)
                        # 這保證了絕對不會重疊，而不是只增加成本
                        if (nr, nc) in idle_obstacles and (nr, nc) != start:
                            continue 

                    step_cost = MOVE_COST
                    
                    # 4. 貨架與排隊區檢查
                    is_physically_occupied = ((nr, nc) in shelf_occ)
                    is_claimed = ((nr, nc) in claimed)
                    
                    if is_physically_occupied or is_claimed:
                        if (nr, nc) == goal or (nr, nc) == start: pass
                        else:
                            step_cost += OCCUPIED_SHELF_COST
                            if is_loaded: step_cost += LARGE_PENALTY
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
                    # 這裡的 has_conflict 檢查可以保留，作為雙重確認
                    if pos in idle_obstacles and pos != start and pos != goal:
                        has_conflict = True
                    path.append((pos, t))
                    curr = came_from[curr]
                path.append((start, start_time))
                path.reverse()
                return path, path[-1][1], final_node[2], has_conflict
            return None, None, None, False
class MapAnalyzer:
    """
    負責地圖的戰略分析：
    1. 計算每個格子的「連通度」(Degree)，判斷是死巷、走道還是路口。
    2. 尋找最近的「安全集結點」(Staging Point)。
    """
    def __init__(self, grid):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.connectivity_map = self._build_connectivity_map()

    def _build_connectivity_map(self):
        # 計算每個空格周圍有多少個可行走的鄰居
        c_map = {}
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == -1: continue # 牆壁
                neighbors = 0
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] != -1:
                        neighbors += 1
                c_map[(r, c)] = neighbors
        return c_map

    def find_safe_buffer(self, start_pos, occupied_set, claimed_set):
        """
        尋找最佳的放貨點 (Cold Zone)。
        優先找連通度低 (<=2) 的點，且不在主幹道上。
        """
        q = deque([start_pos])
        visited = {start_pos}
        
        best_candidate = None
        
        while q:
            curr = q.popleft()
            
            # 檢查是否為合法的空位
            if self.grid[curr[0]][curr[1]] == 1: # 必須是空格
                if curr not in occupied_set and curr not in claimed_set:
                    degree = self.connectivity_map.get(curr, 4)
                    
                    if degree <= 1: return curr # 完美死角
                    if degree == 2 and not best_candidate: best_candidate = curr # 備選走廊
            
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = curr[0]+dr, curr[1]+dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if (nr, nc) not in visited and self.grid[nr][nc] != -1:
                        visited.add((nr, nc))
                        q.append((nr, nc))
        
        return best_candidate if best_candidate else start_pos

    def find_staging_point(self, blockage_pos):
            """
            尋找最近的「巷口集結點」(Staging Point)。
            """
            q = deque([(blockage_pos, 0)])
            visited = {blockage_pos}
            
            while q:
                curr, dist = q.popleft()
                
                degree = self.connectivity_map.get(curr, 0)
                
                # 條件：必須是路口(>=3) 且距離障礙物至少 2 格以上 (避免堵門)
                if degree >= 3 and dist >= 2:
                    return curr
                
                # 如果地圖很窄沒有路口，找一個距離 5 格遠的地方
                if dist >= 5 and self.grid[curr[0]][curr[1]] == 1: 
                    return curr

                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nr, nc = curr[0]+dr, curr[1]+dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        if (nr, nc) not in visited and self.grid[nr][nc] != -1:
                            visited.add((nr, nc))
                            # [注意] 這裡有兩層括號：外層是 append 的，內層是 tuple 的
                            q.append( ((nr, nc), dist + 1) )
            return blockage_pos

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
        self._init_slots()

    def _init_slots(self):
        # 地圖上用 4 代表排隊區
        QUEUE_MARKER = 4
        
        for sid, info in self.stations.items():
            center_pos = info['pos']
            valid_slots = []
            
            # --- V16.1 修改開始: 優先尋找地圖上的 '4' ---
            found_marker_slots = []
            
            # 使用 BFS 擴散尋找離工作站最近的區域
            # 搜尋半徑設為 10，避免抓到隔壁工作站的 4
            max_search_dist = 10 
            
            q = deque([center_pos])
            visited = {center_pos}
            
            while q:
                curr = q.popleft()
                r, c = curr
                
                # 距離檢查
                dist = abs(r - center_pos[0]) + abs(c - center_pos[1])
                if dist > max_search_dist:
                    continue
                
                # 1. 如果這一格是 4，加入候選 (且不是工作站中心本身)
                if self.grid[r][c] == QUEUE_MARKER and curr != center_pos:
                    found_marker_slots.append(curr)
                
                # 2. 繼續擴散
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in visited:
                        # 只走空地(0)、排隊區(4)或工作站本身，不穿過障礙物(-1)
                        if self.grid[nr][nc] != -1: 
                             visited.add((nr, nc))
                             q.append((nr, nc))
            
            if found_marker_slots:
                # 按照離中心距離排序，優先填近的
                found_marker_slots.sort(key=lambda p: abs(p[0]-center_pos[0]) + abs(p[1]-center_pos[1]))
                valid_slots = found_marker_slots[:self.capacity]
            else:
                # --- Fallback: 如果地圖上沒畫 4，維持舊邏輯 (找空位 0) ---
                # print(f"⚠️ 警告: 工作站 {sid} 附近沒找到標記 '{QUEUE_MARKER}'，使用自動找空位邏輯")
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
            # --- V16.1 修改結束 ---

            self.slots_map[sid] = valid_slots
            
            # 出口點邏輯 (維持不變)
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

    # 請加在 PhysicalZoneManager 類別的最後面
    def print_status(self, floor_name):
        print(f"   🏭 [{floor_name} Workstations Report]")
        active_count = 0
        for sid, slots in self.assignments.items():
            used = len(slots)
            limit = self.capacity
            inbound = self.inbound_counts[sid]
            # 顯示格式: StationID: 使用中格子/總格子 (正在路上的車數)
            status_bar = "🟢 Idle" if used == 0 else "🟠 Busy" if used < limit else "🔴 Full"
            print(f"      {status_bar} {sid}: Slots={used}/{limit} | Inbound={inbound}")
            if used > 0: active_count += 1
        if active_count == 0:
            print("      💤 All stations are currently idle.")

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

# ---------------- 主模擬器 V16.0 Optimized ----------------

class SimulationRunner:
    def __init__(self):
        print(f"🚀 [Core V16.0 OPT] 啟動模擬 (Performance Tuned)...")
        self._load_data()
        self.reservations = {'2F': defaultdict(set), '3F': defaultdict(set)}
        self.edge_reservations = {'2F': defaultdict(set), '3F': defaultdict(set)}
        self.shelf_occupancy = {'2F': set(), '3F': set()}
        self.claimed_spots = {'2F': set(), '3F': set()}
        self.pos_to_sid = {'2F': {}, '3F': {}}
        self._init_shelves()
        self.agv_state = self._init_agvs()
        self.agv_tasks = {}

        # [新增區塊 Start] -----------------------------------------
        self.map_analyzer = {
            '2F': MapAnalyzer(self.grid_2f),
            '3F': MapAnalyzer(self.grid_3f)
        }
        self.collaborative_locks = {} # 記錄 {blocker_id: True}
        
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
        self.rescue_queue = {'2F': deque(), '3F': deque()}
        self.rescue_locks = set()
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

    def _init_shelves(self):
        for sid, info in self.shelf_coords.items():
            f, p = info['floor'], info['pos']
            if f == '2F' and self.grid_2f[p[0]][p[1]] != -1: 
                self.shelf_occupancy['2F'].add(p); self.pos_to_sid['2F'][p] = sid
            elif f == '3F' and self.grid_3f[p[0]][p[1]] != -1: 
                self.shelf_occupancy['3F'].add(p); self.pos_to_sid['3F'][p] = sid

    def _init_agvs(self):
        states = {'2F': {}, '3F': {}}
        count_2f = 66
        count_3f = 66
        spots_2f = [p for p in self.valid_spots['2F'] if p not in self.shelf_occupancy['2F']]
        spots_3f = [p for p in self.valid_spots['3F'] if p not in self.shelf_occupancy['3F']]
        if len(spots_2f) < count_2f: spots_2f = self.valid_spots['2F']
        if len(spots_3f) < count_3f: spots_3f = self.valid_spots['3F']
        seed_2f = random.sample(spots_2f, count_2f)
        seed_3f = random.sample(spots_3f, count_3f)
        for i in range(count_2f): 
            states['2F'][i+1] = {'time': 0, 'pos': seed_2f[i], 'dir': 4, 'status': 'IDLE', 'battery': 100, 'force_park': False, 'force_yield': False}
        for i in range(count_3f): 
            states['3F'][i+101] = {'time': 0, 'pos': seed_3f[i], 'dir': 4, 'status': 'IDLE', 'battery': 100, 'force_park': False, 'force_yield': False}
        return states

    def to_dt(self, sec): return self.base_time + timedelta(seconds=sec)

    def _lock_spot(self, floor, pos, start_t, duration):
        end_t = start_t + duration
        for t in range(int(start_t), int(end_t) + 1):
            self.reservations[floor][t].add(pos)

    def _execute_move(self, floor, agv_id, path, type_desc, info_text=""):
        if not path: return
        # Pre-lookup
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

    def _check_line_blockage(self, floor, target_pos):
        r, c = target_pos
        shelf_occ = self.shelf_occupancy[floor]
        for check_c in range(2, c):
            check_pos = (r, check_c)
            if check_pos in shelf_occ:
                return check_pos 
        return None

    def _debug_print_grid(self, floor, center_pos, radius=3):
            r_center, c_center = center_pos
            grid = self.grid_2f if floor == '2F' else self.grid_3f
            shelf_occ = self.shelf_occupancy[floor]
            
            # 建立 AGV 位置查表
            agv_pos_map = {}
            for aid, s in self.agv_state[floor].items():
                agv_pos_map[s['pos']] = aid
                
            print(f"\n🔍 [DEBUG VIEW] Environment around {center_pos} ({floor})")
            print("   " + "".join([f"{c%10}" for c in range(c_center-radius, c_center+radius+1)]))
            
            for r in range(r_center - radius, r_center + radius + 1):
                row_str = f"{r:2d} "
                for c in range(c_center - radius, c_center + radius + 1):
                    # 邊界檢查
                    if r < 0 or r >= 32 or c < 0 or c >= 61:
                        row_str += " "
                        continue
                    
                    pos = (r, c)
                    char = "."
                    
                    # 1. 基礎地圖
                    val = grid[r][c]
                    if val == -1: char = "█" # 牆壁
                    elif val == 1: char = "_" # 空貨架位
                    elif val == 4: char = "≡" # 排隊區
                    
                    # 2. 物件覆蓋
                    if pos in shelf_occ:
                        char = "S" # 這裡有貨架 (Shelf)
                    
                    if pos in agv_pos_map:
                        char = "A" # 這裡有車 (AGV)
                        
                    if pos == center_pos:
                        char = "X" # 擋路的核心點
                    
                    row_str += char
                print(row_str)
            print("   [Legend] █:Wall, _:Slot, ≡:Queue, S:Shelf, A:AGV, X:Target\n")

    def _find_smart_buffer_spot(self, floor, center_pos):
            """
            使用 MapAnalyzer 尋找符合戰略的冷區
            """
            occupied = self.shelf_occupancy[floor]
            claimed = self.claimed_spots[floor]
            # 呼叫 MapAnalyzer 的智慧演算法
            return self.map_analyzer[floor].find_safe_buffer(center_pos, occupied, claimed)

    def _find_smart_storage_spot(self, floor, center_pos, occupied_set):
        q = deque([center_pos])
        visited = {center_pos}
        grid = self.grid_2f if floor=='2F' else self.grid_3f
        while q:
            curr = q.popleft()
            if grid[curr[0]][curr[1]] == 1:
                if curr not in occupied_set and \
                   curr not in self.claimed_spots[floor]:
                    return curr
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = curr[0]+dr, curr[1]+dc
                if 0<=nr<32 and 0<=nc<61 and (nr,nc) not in visited:
                    visited.add((nr,nc))
                    q.append((nr,nc))
        return center_pos

    def _find_parking_spot(self, floor, start_pos):
        grid = self.grid_2f if floor == '2F' else self.grid_3f
        rows, cols = grid.shape
        q = deque([start_pos])
        visited = {start_pos}
        max_steps = 300
        steps = 0
        while q and steps < max_steps:
            curr = q.popleft()
            steps += 1
            if grid[curr[0]][curr[1]] == 1: return curr
            moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            random.shuffle(moves)
            for dr, dc in moves:
                nr, nc = curr[0] + dr, curr[1] + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if (nr, nc) not in visited and grid[nr][nc] != -1 and grid[nr][nc] != 2:
                        visited.add((nr, nc))
                        q.append((nr, nc))
        return start_pos
    
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

    def get_static_obstacles(self, floor, current_sim_time):
            obstacles = set()
            for aid, s in self.agv_state[floor].items():
                # 移除所有 if s['time'] <= ... 的判斷
                # 只要這台車存在，它的位置就是障礙物
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

    # ---------------- 執行邏輯 ----------------

    def run(self):
            # 初始化工作站座標集合，用於 A* 判斷
            station_spots_2f = {info['pos'] for info in self.stations.values() if info['floor'] == '2F'}
            station_spots_3f = {info['pos'] for info in self.stations.values() if info['floor'] == '3F'}
            
            # 初始化 A* 實例
            astars = {
                '2F': TimeAwareAStar(self.grid_2f, self.reservations['2F'], self.edge_reservations['2F'], self.shelf_occupancy['2F'], self.claimed_spots['2F'], '2F', station_spots_2f),
                '3F': TimeAwareAStar(self.grid_3f, self.reservations['3F'], self.edge_reservations['3F'], self.shelf_occupancy['3F'], self.claimed_spots['3F'], '3F', station_spots_3f)
            }
            
            # 將任務依工作站分組
            task_queues = {'2F': defaultdict(deque), '3F': defaultdict(deque)}
            for f in ['2F', '3F']:
                while self.queues[f]:
                    t = self.queues[f].popleft()
                    sid = t['stops'][0]['station']
                    task_queues[f][sid].append(t)

            # 寫入初始位置日誌
            for floor in ['2F', '3F']:
                for aid, s in self.agv_state[floor].items():
                    self.event_writer.writerow([
                        self.to_dt(0), self.to_dt(1), floor, f"AGV_{aid}",
                        s['pos'][1], s['pos'][0], s['pos'][1], s['pos'][0],
                        'INITIAL', 'InitPos'
                    ])

            print("🚦 進入主循環 (包含 Ghost Mode 與 嚴格排隊邏輯)...")
            active_agvs = list(self.agv_state['2F'].keys()) + list(self.agv_state['3F'].keys())
            sim_time = 0
            done_count = 0
            global_pbar = 0
            
            while True:
                global_pbar += 1
                if global_pbar % 20 == 0:
                    print(f"⏱ Loop {global_pbar} | Done: {done_count}")
                    self.zm['2F'].print_status('2F')
                    self.zm['3F'].print_status('3F')
                    print("-" * 50)

                    rem_tasks = sum([len(q) for f in task_queues for q in task_queues[f].values()])
                    rem_rescue = len(self.rescue_queue['2F']) + len(self.rescue_queue['3F'])
                    active_working = len([a for a in active_agvs if self.agv_tasks.get(a)])
                    if rem_tasks == 0 and rem_rescue == 0 and active_working == 0:
                        break

                # 依時間排序 AGV，確保模擬時間軸同步
                all_agvs_sorted = sorted(active_agvs, key=lambda aid: self.agv_state['2F' if aid < 100 else '3F'][aid]['time'])
                
                for agv_id in all_agvs_sorted:
                    floor = '2F' if agv_id < 100 else '3F'
                    state = self.agv_state[floor][agv_id]
                    astar = astars[floor]
                    
                    # 時間同步控制：避免跑太快
                    if state['time'] > sim_time + 300: continue
                    if state['time'] > sim_time: sim_time = state['time']

                    curr_status = state['status']
                    curr_pos = state['pos']
                    curr_time = state['time']

                    # 取得當前樓層的靜態障礙車集合
                    current_idle_obstacles = self.get_static_obstacles(floor, sim_time)
                    
                    # --- 狀態 1: IDLE (閒置/接單/避讓) ---
                    if curr_status == 'IDLE':
                        # [協同邏輯 1] 檢查是否正在巷口等待解鎖
                        if state.get('waiting_for_clearance'):
                            target_blocker = state['waiting_for_clearance']
                            blocker_pos = self.shelf_coords[target_blocker]['pos']
                            
                            # 如果障礙物已經被移走 (不在原位了) 或者 協同鎖已被移除
                            if blocker_pos not in self.shelf_occupancy[floor] or target_blocker not in self.collaborative_locks:
                                print(f"🚦 [GO] 障礙物 {target_blocker} 已清除，工作車 AGV_{agv_id} 衝啊！")
                                state['waiting_for_clearance'] = None # 解除等待，下一幀會自動接任務
                            else:
                                state['time'] += 2 # 繼續在巷口等
                                self._lock_spot(floor, curr_pos, curr_time, 2)
                            continue

                        # [Yield 邏輯保持不變]
                        if state.get('force_yield'):
                            yield_spot = self._find_yield_spot(floor, curr_pos)
                            if yield_spot and yield_spot != curr_pos:
                                path_yield, _, _, _ = astar.find_path(curr_pos, yield_spot, curr_time, check_only=False)
                                if path_yield:
                                    self._execute_move(floor, agv_id, path_yield, 'YIELD', 'SmartYield')
                                    state['status'] = 'IDLE' 
                                    state['force_yield'] = False
                                    continue 
                            state['force_yield'] = False 

                        # [救援任務領取邏輯]
                        if self.rescue_queue[floor]:
                            # 檢查自己是否離救援點夠近 (簡易最佳化)，這裡先直接領取
                            rescue_task = self.rescue_queue[floor].popleft()
                            self.agv_tasks[agv_id] = rescue_task
                            state['status'] = 'RESCUE_MODE'
                            continue

                        best_task = None
                        candidate_stations = list(task_queues[floor].keys())
                        random.shuffle(candidate_stations)
                        
                        for sid in candidate_stations:
                            q = task_queues[floor][sid]
                            if not q: continue
                            if not self.zm[floor].can_add_inbound(sid): continue
                            
                            task = q[0]
                            shelf_id = task['shelf_id']
                            if shelf_id in self.rescue_locks: continue # 別人正在救，跳過
                            
                            shelf_pos = self.shelf_coords[shelf_id]['pos']
                            station_pos = self.stations[sid]['pos']

                            # 嘗試尋路
                            path1, _, _, conflict1 = astar.find_path(curr_pos, shelf_pos, curr_time, idle_obstacles=current_idle_obstacles, check_only=True)
                            
                            if not path1:
                                # [協同邏輯 2] 路徑不通，檢查是否需要雙車戰術
                                blocker_pos = self._check_line_blockage(floor, shelf_pos)
                                
                                if blocker_pos:
                                    blocker_sid = self.pos_to_sid[floor].get(blocker_pos)
                                    # 只有當這個障礙物還沒被鎖定時，才發起救援
                                    if blocker_sid and blocker_sid not in self.collaborative_locks:
                                        print(f"🚨 [協同救援] {blocker_sid} 擋路，啟動雙車戰術！(Worker: {agv_id})")
                                        self._debug_print_grid(floor, blocker_pos, radius=3)

                                        # 1. 透過 MapAnalyzer 找戰略點
                                        safe_buffer = self._find_smart_buffer_spot(floor, blocker_pos) # 絕對冷區
                                        staging_point = self.map_analyzer[floor].find_staging_point(blocker_pos) # 安全巷口
                                        
                                        # 2. 鎖定並發布救援任務
                                        self.collaborative_locks[blocker_sid] = True
                                        self.rescue_locks.add(blocker_sid)
                                        self.rescue_queue[floor].append({
                                            'type': 'RESCUE', 
                                            'shelf_id': blocker_sid,
                                            'target_buffer': safe_buffer # 指定落點
                                        })
                                        
                                        # 3. 指派自己(工作車)去巷口等待
                                        path_staging, _, _, _ = astar.find_path(curr_pos, staging_point, curr_time, idle_obstacles=current_idle_obstacles)
                                        
                                        if path_staging:
                                            self._execute_move(floor, agv_id, path_staging, 'TACTICAL_WAIT', f"Wait@{staging_point}")
                                            state['waiting_for_clearance'] = blocker_sid
                                            state['status'] = 'IDLE' # 到了之後會變回 IDLE，被上面的邏輯接手
                                        else:
                                            # 去不了巷口，只好原地等
                                            state['waiting_for_clearance'] = blocker_sid
                                            state['time'] += 5
                                            self._lock_spot(floor, curr_pos, curr_time, 5)
                                        continue # 結束這回合
                                
                                # 原本的 Ghost Path 邏輯 (保留作為最後防線)
                                state['time'] += 5
                                self._lock_spot(floor, curr_pos, curr_time, 5)
                                continue 
                            
                            if conflict1: 
                                self.resolve_idle_conflict(floor, path1, current_idle_obstacles)
                                state['time'] += 5
                                self._lock_spot(floor, curr_pos, curr_time, 5)
                                continue

                            # ... (原本的 path2 檢查與任務領取邏輯，保持不變) ...
                            est_pickup_time = curr_time + len(path1) + 10
                            path2, _, _, conflict2 = astar.find_path(shelf_pos, station_pos, est_pickup_time, idle_obstacles=current_idle_obstacles, is_loaded=True, check_only=True)
                            if not path2: continue
                            if conflict2: 
                                self.resolve_idle_conflict(floor, path2, current_idle_obstacles)
                                state['time'] += 5
                                self._lock_spot(floor, curr_pos, curr_time, 5)
                                continue

                            best_task = q.popleft()
                            best_sid = sid
                            self.zm[floor].register_inbound(sid) 
                            break
                        
                        if best_task:
                            self.agv_tasks[agv_id] = best_task
                            state['status'] = 'MOVING_TO_PICK'
                        else:
                            # ... (原本的 AutoPark 邏輯保持不變) ...
                            need_to_park = state['force_park'] or (random.random() < 0.05)
                            path_park = None
                            if need_to_park:
                                parking_spot = self._find_parking_spot(floor, curr_pos)
                                if parking_spot != curr_pos:
                                    path_park, _, _, conflict_p = astar.find_path(
                                        curr_pos, parking_spot, curr_time, 
                                        idle_obstacles=current_idle_obstacles
                                    )
                                    if conflict_p: path_park = None
                            
                            if path_park:
                                reason = 'Yield' if state['force_park'] else 'AutoPark'
                                self._execute_move(floor, agv_id, path_park, 'PARKING', reason)
                                state['status'] = 'IDLE'
                                state['force_park'] = False
                            else:
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
                            state['time'] += 5
                            self._lock_spot(floor, curr_pos, curr_time, 5)

                    # --- 狀態 3: LOADED (搬貨去工作站) ---
                    elif curr_status == 'LOADED':
                        task = self.agv_tasks[agv_id]
                        sid = task['stops'][0]['station']
                        st_center = self.stations[sid]['pos']
                        dist = abs(curr_pos[0] - st_center[0]) + abs(curr_pos[1] - st_center[1])
                        
                        target_dest = None
                        is_final_approach = False
                        
                        # 距離 <= 3 時才申請排隊位
                        if dist <= 3:
                            slot_pos = self.zm[floor].assign_spot(sid, agv_id)
                            if slot_pos:
                                target_dest = slot_pos
                                is_final_approach = True
                            else:
                                # 沒位子，原地等待 (這在嚴格限流下發生機率會降低)
                                state['time'] += 5
                                self._lock_spot(floor, curr_pos, curr_time, 5)
                                continue
                        else:
                            target_dest = st_center
                            is_final_approach = False
                        
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
                                # 到達排隊位，執行作業
                                proc_time = task['stops'][0]['time']
                                state['time'] += proc_time
                                self._lock_spot(floor, target_dest, state['time'] - proc_time, proc_time)

                                print(f"⚙️ [Work] AGV_{agv_id} 在 {sid} 進行作業 (耗時 {proc_time}s)...")
                                
                                # KPI 記錄
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
                            state['time'] += 5
                            self._lock_spot(floor, curr_pos, curr_time, 5)

                    # --- 狀態 4: RETURNING (嚴格物理模式 - No Ghost Mode) ---
                    elif curr_status == 'RETURNING':
                        task = self.agv_tasks[agv_id]
                        shelf_id = task['shelf_id']
                        orig_pos = self.shelf_coords[shelf_id]['pos']
                        sid = task['stops'][0]['station']
                        
                        # 計算中途點 (出口點)
                        exit_pt = self.zm[floor].exit_points.get(sid)
                        dist_to_exit = 999
                        if exit_pt:
                            dist_to_exit = abs(curr_pos[0] - exit_pt[0]) + abs(curr_pos[1] - exit_pt[1])
                        
                        # 決定要放回哪裡
                        target_drop = orig_pos
                        # 如果原位被佔，找冷區
                        if target_drop in self.shelf_occupancy[floor]: 
                            target_drop = self._find_smart_buffer_spot(floor, orig_pos)
                        
                        current_target = target_drop
                        
                        # 如果在工作站附近，先去出口點 (避免直接穿越工作站)
                        if exit_pt and dist_to_exit > 2 and dist_to_exit < 20: 
                            st_dist = abs(curr_pos[0] - self.stations[sid]['pos'][0]) + abs(curr_pos[1] - self.stations[sid]['pos'][1])
                            if st_dist < 8:
                                current_target = exit_pt
                        
                        # 暫時鎖定目標點 (這是為了 A* 尋路時判定目標點是合法的)
                        self.claimed_spots[floor].add(current_target)
                        
                        # === 嚴格物理尋路 (Strict Physics) ===
                        # 這裡移除了 ignore_others=True 的邏輯
                        # 也就是說：如果路被擋住，path 就會是 None，車子會被迫停下來
                        path, end_t, _, conflict = astar.find_path(
                            curr_pos, current_target, curr_time, 
                            idle_obstacles=current_idle_obstacles,
                            is_loaded=True,
                            ignore_others=False # [重要] 強制檢查動態障礙物，絕不穿透
                        )
                        
                        if path:
                            # 如果路徑上有靜態 IDLE 車輛，嘗試叫它讓路
                            if conflict: 
                                self.resolve_idle_conflict(floor, path, current_idle_obstacles)
                                self.claimed_spots[floor].remove(current_target) 
                                state['time'] += 5
                                self._lock_spot(floor, curr_pos, curr_time, 5)
                                continue

                            # 執行移動
                            self._execute_move(floor, agv_id, path, 'AGV_MOVE')
                            
                            # 如果只是到了出口點，下一輪繼續算回程
                            if current_target == exit_pt:
                                self.claimed_spots[floor].remove(current_target)
                                state['time'] += 2
                                continue
                            
                            # 到達最終儲位：卸貨
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
                            # [路徑被擋住]
                            # 因為移除了 Ghost Mode，這裡不會強行穿透
                            # 而是釋放目標鎖定，原地等待，下一幀重試
                            self.claimed_spots[floor].remove(current_target)
                            
                            # 檢查是不是被貨架擋住 (觸發救援的機會)
                            # 注意：RETURNING 狀態通常不觸發救援，避免邏輯過於複雜
                            # 我們假設 Co-Op 救援機制會在 IDLE 車輛那邊解決這個問題
                            
                            state['time'] += 5
                            self._lock_spot(floor, curr_pos, curr_time, 5)
                            # print(f"🚫 AGV_{agv_id} 回程受阻，原地等待...")
                    # --- 狀態 5: RESCUE_MODE (移動貨架以解鎖) ---
                    elif curr_status == 'RESCUE_MODE':
                        task = self.agv_tasks[agv_id]
                        target_sid = task['shelf_id']
                        target_pos = self.shelf_coords[target_sid]['pos']
                        
                        # [協同邏輯 3] 讀取任務中指定的安全冷區
                        designated_buffer = task.get('target_buffer')
                        
                        if target_pos in self.shelf_occupancy[floor]:
                            # 第一階段：去拿障礙物
                            path, end_t, _, conflict = astar.find_path(curr_pos, target_pos, curr_time, idle_obstacles=current_idle_obstacles, is_loaded=False)
                            
                            if path:
                                if conflict: 
                                    self.resolve_idle_conflict(floor, path, current_idle_obstacles)
                                    state['time'] += 5
                                    self._lock_spot(floor, curr_pos, curr_time, 5)
                                    continue

                                self._execute_move(floor, agv_id, path, 'AGV_MOVE', 'RescueApproach')
                                self.shelf_occupancy[floor].remove(target_pos)
                                self.event_writer.writerow([self.to_dt(end_t), self.to_dt(end_t+5), floor, f"AGV_{agv_id}", target_pos[1], target_pos[0], target_pos[1], target_pos[0], 'SHUFFLE_LOAD', f"{target_sid}"])
                                state['time'] += 5
                                
                                # 第二階段：搬去冷區 (使用指定點 或 重新尋找)
                                safe_spot = designated_buffer if designated_buffer else self._find_smart_buffer_spot(floor, target_pos)
                                
                                self.claimed_spots[floor].add(safe_spot)
                                
                                path2, end_t2, _, conflict2 = astar.find_path(target_pos, safe_spot, state['time'], idle_obstacles=current_idle_obstacles, is_loaded=True)
                                
                                if path2:
                                    if conflict2: 
                                        self.resolve_idle_conflict(floor, path2, current_idle_obstacles)
                                        self.claimed_spots[floor].remove(safe_spot)
                                        state['time'] += 5
                                        self._lock_spot(floor, target_pos, state['time']-5, 5)
                                        continue
                                    
                                    self._execute_move(floor, agv_id, path2, 'AGV_MOVE', 'RescueToColdZone')
                                    self.shelf_occupancy[floor].add(safe_spot)
                                    self.claimed_spots[floor].remove(safe_spot)
                                    self.shelf_coords[target_sid]['pos'] = safe_spot
                                    self.pos_to_sid[floor][safe_spot] = target_sid
                                    
                                    self.event_writer.writerow([self.to_dt(end_t2), self.to_dt(end_t2+5), floor, f"AGV_{agv_id}", safe_spot[1], safe_spot[0], safe_spot[1], safe_spot[0], 'SHUFFLE_UNLOAD', f"{target_sid}"])
                                    state['time'] += 5
                                    
                                    # [協同邏輯 4] 任務完成，解鎖！工作車會收到訊號
                                    if target_sid in self.collaborative_locks:
                                        del self.collaborative_locks[target_sid]
                                        print(f"🔓 [UNLOCK] {target_sid} 已移至安全區 {safe_spot}，解鎖！")
                                    
                                    if target_sid in self.rescue_locks: self.rescue_locks.remove(target_sid)
                                    state['status'] = 'IDLE'
                                    del self.agv_tasks[agv_id]
                                else:
                                    # 去不了冷區，釋放並重試
                                    self.claimed_spots[floor].remove(safe_spot)
                                    state['status'] = 'RESCUE_MODE'
                            else:
                                state['time'] += 5
                                self._lock_spot(floor, curr_pos, curr_time, 5)
                        else:
                            # 異常狀況：障礙物不見了，直接完成
                            state['status'] = 'IDLE'
                            del self.agv_tasks[agv_id]
                            if target_sid in self.collaborative_locks: del self.collaborative_locks[target_sid]
                            if target_sid in self.rescue_locks: self.rescue_locks.remove(target_sid)

            self.event_writer.close()
            self.kpi_writer.close()
            print("🎉 V16.2 Optimized 模擬結束")

if __name__ == "__main__":
    SimulationRunner().run()
