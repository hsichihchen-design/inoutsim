import pandas as pd
import numpy as np
import os
import time
import heapq
import csv
import random
import math
from collections import defaultdict, deque, Counter
from datetime import datetime, timedelta

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
# ----------------------------------------

class BatchWriter:
    def __init__(self, filepath, header, chunk_size=20000):
        self.f = open(filepath, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.f)
        self.writer.writerow(header)
        self.buffer = []
        self.chunk_size = chunk_size
    
    def writerow(self, row):
        self.buffer.append(row)
        if len(self.buffer) >= self.chunk_size:
            self.flush()
            
    def flush(self):
        if self.buffer:
            self.writer.writerows(self.buffer)
            self.buffer = []
            
    def close(self):
        self.flush()
        self.f.close()

class TimeAwareAStar:
    """
    V2.0: 引入方向性 (Directional) 與 轉向成本 (Turning Cost)
    同時支援 邊緣鎖定 (Edge Constraints) 防止對向穿模
    """
    def __init__(self, grid, reservations_dict, edge_reservations_dict, shelf_occupancy_set):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.reservations = reservations_dict
        self.edge_reservations = edge_reservations_dict # key: time, val: set of ((r_from, c_from), (r_to, c_to))
        self.shelf_occupancy = shelf_occupancy_set 
        
        # 定義移動與方向索引: 0:Right, 1:Left, 2:Down, 3:Up, 4:Wait/Start
        self.moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)] 
        self.DIR_MAP = {(0, 1): 0, (0, -1): 1, (1, 0): 2, (-1, 0): 3, (0, 0): 4}
        
        # 成本參數
        self.NORMAL_COST = 1.0
        self.TURNING_COST = 3.0    # 加大轉彎成本，減少蛇行
        self.U_TURN_COST = 5.0     # 極力避免掉頭
        self.WAIT_COST = 1.0
        self.TUNNEL_COST = 50.0

        self.hard_limit = 2000 # 搜尋步數上限

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start, goal, start_time_sec, static_blockers=None, is_loaded=False, ignore_dynamic=False, allow_tunneling=False):
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols): return None, None
        if not (0 <= goal[0] < self.rows and 0 <= goal[1] < self.cols): return None, None
        if self.grid[start[0]][start[1]] == -1: return None, None
        if self.grid[goal[0]][goal[1]] == -1: return None, None

        if start == goal: return [(start, start_time_sec)], start_time_sec
        
        dist = self.heuristic(start, goal)
        max_steps = 5000 if ignore_dynamic else min(1000, max(200, dist * 20))
        
        HEURISTIC_WEIGHT = 1.5 

        open_set = []
        h_start = self.heuristic(start, goal)
        
        # Heap State: (F, H, Time, Pos, Direction_Index)
        # 初始方向設為 4 (None/Wait)
        start_dir = 4
        heapq.heappush(open_set, (h_start, h_start, start_time_sec, start, start_dir))
        
        # G_Score Key: (Pos, Time, Direction) -> 必須包含方向，因為不同方向抵達同一點的成本不同
        g_score = {}
        g_score[(start, start_time_sec, start_dir)] = 0
        
        came_from = {}
        steps_count = 0
        
        while open_set:
            steps_count += 1
            if steps_count > max_steps: break 
            
            f, h, current_time, current, current_dir = heapq.heappop(open_set)

            if current == goal:
                return self._reconstruct_path(came_from, (current, current_time, current_dir), start, start_time_sec)

            curr_g = g_score.get((current, current_time, current_dir), float('inf'))
            if curr_g < (f - h * HEURISTIC_WEIGHT): continue

            for move_idx, (dr, dc) in enumerate(self.moves):
                nr, nc = current[0] + dr, current[1] + dc
                next_time = current_time + 1 
                
                # 1. 邊界與牆壁檢查
                if not (0 <= nr < self.rows and 0 <= nc < self.cols): continue
                if self.grid[nr][nc] == -1: continue 
                
                # 2. 動態障礙檢查 (Dynamic Obstacles)
                if not ignore_dynamic:
                    if (next_time - start_time_sec) < 60: # 只看未來 60 秒
                        # A. 點鎖定 (Vertex Constraint)
                        if next_time in self.reservations and (nr, nc) in self.reservations[next_time]:
                            continue
                        
                        # B. 邊鎖定 (Edge Constraint) - 防止對向交換
                        # 如果我要從 Current -> Next，不能有人在同一時間從 Next -> Current
                        reverse_edge = ((nr, nc), current)
                        if current_time in self.edge_reservations and reverse_edge in self.edge_reservations[current_time]:
                            continue

                # 3. 靜態料架檢查 (Static Shelf Logic)
                is_spot_occupied = ((nr, nc) in self.shelf_occupancy)
                step_cost = self.NORMAL_COST

                if is_loaded:
                    if is_spot_occupied:
                        if (nr, nc) == goal or (nr, nc) == start: pass 
                        elif allow_tunneling: step_cost += self.TUNNEL_COST
                        else: continue # 載貨且不能鑽，視為牆壁
                else:
                    if is_spot_occupied: step_cost += 0.5 # 空車鑽過貨架稍微慢一點點

                # 4. 轉向成本計算 (Turning Cost)
                new_dir = move_idx
                if current_dir != 4: # 排除剛起步
                    if new_dir == 4: # 等待
                         step_cost += self.WAIT_COST
                    elif new_dir != current_dir:
                        # 檢查是否為掉頭 (U-Turn)
                        # 利用 vector 相加是否為 (0,0) 判斷反向 (僅適用於 UDLR)
                        vec_sum = (self.moves[current_dir][0] + dr, self.moves[current_dir][1] + dc)
                        if vec_sum == (0,0):
                            step_cost += self.U_TURN_COST
                        else:
                            step_cost += self.TURNING_COST
                
                new_g = curr_g + step_cost
                
                state_key = ((nr, nc), next_time, new_dir)
                
                if state_key not in g_score or new_g < g_score[state_key]:
                    g_score[state_key] = new_g
                    h = self.heuristic((nr, nc), goal)
                    f = new_g + (h * HEURISTIC_WEIGHT)
                    heapq.heappush(open_set, (f, h, next_time, (nr, nc), new_dir))
                    came_from[state_key] = (current, current_time, current_dir)
                        
        return None, None

    def _reconstruct_path(self, came_from, current_node, start_pos, start_time):
        path = []
        curr = current_node
        while curr in came_from:
            pos, t, d = curr
            path.append((pos, t))
            curr = came_from[curr]
        path.append((start_pos, start_time))
        path.reverse()
        return path, path[-1][1]

class TrafficController:
    """
    簡化版交通控制器
    主要負責: 
    1. 讓路 (Yielding) - 當路徑被暫時擋住時
    2. 倒車 (Backtracking) - 當死路時
    """
    def __init__(self, grid, agv_state_pool, reservations):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.agv_pool = agv_state_pool 
        self.reservations = reservations

    def clear_path_obstacles(self, start_pos, goal_pos, current_time, w_evt, floor, my_agv_name):
        # 簡單嘗試找避難點讓路，若失敗則回傳 False
        # 這裡不處理複雜的 Rescue，只處理短暫的 AGV 互擋
        blocker_id = None
        blocker_pos = None
        
        # 簡易視線檢查 (Raycast)
        curr = list(start_pos)
        target = list(goal_pos)
        steps = 0
        while curr != target and steps < 5:
            if curr[0] < target[0]: curr[0] += 1
            elif curr[0] > target[0]: curr[0] -= 1
            elif curr[1] < target[1]: curr[1] += 1
            elif curr[1] > target[1]: curr[1] -= 1
            check_pos = tuple(curr)
            
            for agv_id, state in self.agv_pool.items():
                if f"AGV_{agv_id}" == my_agv_name: continue
                if state['pos'] == check_pos:
                    blocker_id = agv_id
                    blocker_pos = check_pos
                    break
            if blocker_id: break
            steps += 1
            
        if not blocker_id: return False, 0

        sanctuary = self._find_sanctuary(blocker_pos, current_time)
        if sanctuary:
            # 瞬移讓路 (模擬對方收到讓路指令)
            self.agv_pool[blocker_id]['pos'] = sanctuary
            cost = 5.0
            w_evt.writerow([
                datetime.fromtimestamp(current_time), datetime.fromtimestamp(current_time+int(cost)), 
                floor, f"AGV_{blocker_id}", blocker_pos[1], blocker_pos[0], sanctuary[1], sanctuary[0], 'YIELD', f'Yield for {my_agv_name}'
            ])
            # 鎖定該區域
            for t in range(int(cost) + 5):
                self.reservations[current_time + t].add(sanctuary)
            return True, cost
        return False, 0

    def attempt_backtrack(self, current_pos, goal_pos, current_time, w_evt, floor, agv_name):
        # 尋找最近的一個空位倒車
        best_retreat = None
        max_dist = -1
        
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = current_pos[0]+dr, current_pos[1]+dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.grid[nr][nc] != -1:
                    is_occupied = False
                    for state in self.agv_pool.values():
                        if state['pos'] == (nr, nc): is_occupied = True; break
                    
                    if not is_occupied:
                        # 選擇離目標較遠的格子當作撤退點 (拉開空間)
                        dist_to_goal = abs(nr - goal_pos[0]) + abs(nc - goal_pos[1])
                        if dist_to_goal > max_dist:
                            max_dist = dist_to_goal
                            best_retreat = (nr, nc)
        
        if best_retreat:
            w_evt.writerow([
                datetime.fromtimestamp(current_time), datetime.fromtimestamp(current_time+5), 
                floor, agv_name, current_pos[1], current_pos[0], best_retreat[1], best_retreat[0], 'YIELD', 'Backtracking'
            ])
            for t in range(current_time, current_time+8):
                self.reservations[t].add(best_retreat)
            return True, best_retreat, 5
        return False, current_pos, 0

    def _find_sanctuary(self, start_pos, current_time):
        q = deque([start_pos])
        visited = {start_pos}
        max_search = 50
        count = 0
        while q and count < max_search:
            curr = q.popleft()
            count += 1
            if curr != start_pos and self.grid[curr[0]][curr[1]] != -1:
                is_reserved = False
                for t in range(3): 
                    if curr in self.reservations[current_time + t]:
                        is_reserved = True; break
                if not is_reserved:
                    is_occupied = False
                    for state in self.agv_pool.values():
                        if state['pos'] == curr:
                            is_occupied = True; break
                    if not is_occupied:
                        return curr 
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = curr[0]+dr, curr[1]+dc
                if 0<=nr<self.rows and 0<=nc<self.cols:
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        q.append((nr, nc))
        return None

# [V67] Zone Manager 總量管制 (Capacity -> 4)
class ZoneManager:
    def __init__(self, stations_info, capacity=4): 
        self.zones = {} 
        self.capacity = capacity
        for sid in stations_info:
            self.zones[sid] = 0
            
    def can_enter(self, sid):
        if sid not in self.zones: return False
        return self.zones[sid] < self.capacity
        
    def enter(self, sid):
        if sid in self.zones:
            self.zones[sid] += 1
            return True
        return False
        
    def exit(self, sid):
        if sid in self.zones and self.zones[sid] > 0:
            self.zones[sid] -= 1
            
    def get_usage(self, sid):
        return self.zones.get(sid, 0)
    
    def force_reset(self, sid):
        if sid in self.zones:
            self.zones[sid] = 0

class PhysicalQueueManager:
    def __init__(self, stations_info):
        self.station_queues = {} 
        for sid, info in stations_info.items():
            r, c = info['pos']
            q_slots = []
            for col in range(2, 5): 
                q_slots.append((r, col))
            exits = [(r-1, 1), (r+1, 1)]
            self.station_queues[sid] = {
                'slots': q_slots,
                'exits': exits,
                'occupants': [None] * len(q_slots),
                'processing': None,
                'processing_since': None 
            }

    def get_target_for_agv(self, sid, agv_id):
        q_data = self.station_queues.get(sid)
        if not q_data: return None, False 
        
        if q_data['processing'] == agv_id: return None, True 
            
        if agv_id in q_data['occupants']:
            idx = q_data['occupants'].index(agv_id)
            if idx == 0:
                if q_data['processing'] is None:
                    return (q_data['slots'][0][0], 1), True 
                else:
                    return q_data['slots'][0], False 
            else:
                next_idx = idx - 1
                if q_data['occupants'][next_idx] is None:
                    q_data['occupants'][next_idx] = agv_id
                    # 舊位置的清除由 update_position 處理 (Atomic Handoff)
                    return q_data['slots'][next_idx], False
                else:
                    return q_data['slots'][idx], False 
                    
        for i in range(len(q_data['slots'])-1, -1, -1):
            if q_data['occupants'][i] is None:
                last_slot_idx = len(q_data['slots']) - 1
                if i == last_slot_idx:
                    q_data['occupants'][last_slot_idx] = agv_id 
                    return q_data['slots'][last_slot_idx], False
        return None, False

    def update_position(self, sid, agv_id, current_pos):
        q_data = self.station_queues.get(sid)
        if not q_data: return
        proc_pos = (q_data['slots'][0][0], 1)
        if current_pos == proc_pos:
            if q_data['processing'] != agv_id:
                q_data['processing'] = agv_id
                q_data['processing_since'] = 0 
            if agv_id in q_data['occupants']:
                idx = q_data['occupants'].index(agv_id)
                q_data['occupants'][idx] = None
            return

        if current_pos in q_data['slots']:
            idx = q_data['slots'].index(current_pos)
            for i, occupant in enumerate(q_data['occupants']):
                if occupant == agv_id and i != idx:
                    q_data['occupants'][i] = None
            q_data['occupants'][idx] = agv_id

    def set_processing_time(self, sid, current_time):
        q_data = self.station_queues.get(sid)
        if q_data and q_data['processing'] is not None:
            if q_data['processing_since'] == 0: 
                q_data['processing_since'] = current_time

    def release_station(self, sid, agv_id):
        q_data = self.station_queues.get(sid)
        if q_data and q_data['processing'] == agv_id:
            q_data['processing'] = None
            q_data['processing_since'] = None
            
    def get_exit_spot(self, sid):
        q_data = self.station_queues.get(sid)
        if q_data: return q_data['exits'][0] 
        return None
        
    def is_station_jammed(self, sid, current_time, threshold=300):
        q_data = self.station_queues.get(sid)
        if not q_data: return False
        if q_data['processing'] is not None and q_data['processing_since'] is not None:
            if q_data['processing_since'] > 0: 
                elapsed = current_time - q_data['processing_since']
                if elapsed > threshold: return True
        return False

class OrderProcessor:
    def __init__(self, stations_2f, stations_3f):
        self.stations = {'2F': list(stations_2f.keys()), '3F': list(stations_3f.keys())}
        self.cust_station_map = {'2F': {}, '3F': {}} 

    def process_wave(self, wave_orders, floor):
        wave_custs = wave_orders['PARTCUSTID'].unique()
        available_stations = self.stations.get(floor, [])
        if not available_stations: return []
        floor_map = self.cust_station_map[floor]
        for i, cust_id in enumerate(wave_custs):
            if cust_id not in floor_map:
                st_idx = i % len(available_stations)
                floor_map[cust_id] = available_stations[st_idx]
                
        shelf_tasks = defaultdict(list)
        for _, row in wave_orders.iterrows():
            loc = str(row.get('LOC', '')).strip()
            if len(loc) < 9: continue 
            shelf_id = loc[:9]
            face = loc[10] if len(loc) > 10 else 'A'
            cust_id = row.get('PARTCUSTID')
            target_st = floor_map.get(cust_id)
            if not target_st: target_st = random.choice(available_stations)
            shelf_tasks[shelf_id].append({
                'face': face, 'station': target_st,
                'sku': f"{row.get('FRCD','')}_{row.get('PARTNO','')}",
                'qty': row.get('QTY', 1), 'order_row': row,
                'datetime': row.get('datetime', None) 
            })
            
        final_tasks = []
        for shelf_id, orders in shelf_tasks.items():
            orders.sort(key=lambda x: (x['station'], x['face']))
            stops = []
            current_st = None
            current_face = None
            current_sku_group = defaultdict(int) 
            task_dt = orders[0]['datetime'] 
            for o in orders:
                st = o['station']
                face = o['face']
                sku = o['sku']
                if o['datetime'] < task_dt: task_dt = o['datetime']
                if (st != current_st or face != current_face) and current_st is not None:
                    proc_time = self._calc_time(current_sku_group)
                    stops.append({'station': current_st, 'face': current_face, 'time': proc_time})
                    current_sku_group = defaultdict(int)
                current_st = st
                current_face = face
                current_sku_group[sku] += 1
            if current_st is not None:
                proc_time = self._calc_time(current_sku_group)
                stops.append({'station': current_st, 'face': current_face, 'time': proc_time})
            final_tasks.append({
                'shelf_id': shelf_id, 'stops': stops,
                'wave_id': orders[0]['order_row'].get('WAVE_ID'),
                'raw_orders': [o['order_row'] for o in orders],
                'datetime': task_dt,
                'priority': 0 # Normal Priority
            })
        return final_tasks

    def _calc_time(self, sku_group):
        total_time = 0
        for sku, count in sku_group.items(): total_time += 15 + (count * 5)
        return total_time

class LiveMonitor:
    def __init__(self):
        self.stats = {'Load':0, 'Visit':0, 'Return':0, 'Park':0, 'Rescue':0}
        self.teleports = Counter()
        self.start_time = time.time()
    
    def log_success(self, category):
        self.stats[category] += 1
        
    def log_teleport(self, category, reason):
        self.teleports[f"{category}:{reason}"] += 1
        
    def print_status(self, done_count, total_tasks, agv_pool, rescue_q_len):
        elapsed = time.time() - self.start_time
        active_agvs = sum(1 for s in agv_pool.values() if s['time'] > 0)
        top_errors = self.teleports.most_common(3)
        err_str = " | ".join([f"{k}:{v}" for k,v in top_errors])
        print(f"\n[{elapsed:.0f}s] {done_count}/{total_tasks} | 🚗 Act:{active_agvs} 🔥RescuePending:{rescue_q_len}")
        print(f"   📊 S:{self.stats['Load']}/{self.stats['Visit']}/{self.stats['Return']}/{self.stats['Park']}/🚑{self.stats['Rescue']}")
        print(f"   ⚠️ Err: {err_str}")

class AdvancedSimulationRunner:
    def __init__(self):
        print(f"🚀 [Step 4] 啟動進階模擬 (V75: 轉向成本 + 邊緣鎖定 + 主動救援 + 密度管制)...")
        
        self.grid_2f = self._load_map_correct('2F_map.xlsx', 32, 61)
        self.grid_3f = self._load_map_correct('3F_map.xlsx', 32, 61)
        
        self.reservations_2f = defaultdict(set)
        self.reservations_3f = defaultdict(set)
        
        # [V75] 新增：Edge Reservations (防止對向交換)
        self.edge_reservations_2f = defaultdict(set)
        self.edge_reservations_3f = defaultdict(set)
        
        self.last_clean_time_2f = 0
        self.last_clean_time_3f = 0

        self.shelf_coords = self._load_shelf_coords()
        self.shelf_occupancy = {'2F': set(), '3F': set()}
        self.valid_storage_spots = {'2F': set(), '3F': set()}
        self.pos_to_sid_2f = {}
        self.pos_to_sid_3f = {}
        
        self._init_spots(self.grid_2f, '2F')
        self._init_spots(self.grid_3f, '3F')

        self.inventory_map = self._load_inventory() 
        self.all_tasks_raw = self._load_all_tasks()
        self._assign_locations_smartly(self.all_tasks_raw)
        
        self.stations = self._init_stations()
        st_2f = {k:v for k,v in self.stations.items() if v['floor']=='2F'}
        st_3f = {k:v for k,v in self.stations.items() if v['floor']=='3F'}
        
        self.processor = OrderProcessor(st_2f, st_3f)
        
        self.used_spots_2f = set()
        self.used_spots_3f = set()
        
        self.agv_state = {
            '2F': {i: {'time': (i-1)*15, 'pos': self._get_strict_spawn_spot(self.grid_2f, self.used_spots_2f, '2F')} for i in range(1, 19)},
            '3F': {i: {'time': (i-101)*15, 'pos': self._get_strict_spawn_spot(self.grid_3f, self.used_spots_3f, '3F')} for i in range(101, 119)}
        }
        
        self.traffic_2f = TrafficController(self.grid_2f, self.agv_state['2F'], self.reservations_2f)
        self.traffic_3f = TrafficController(self.grid_3f, self.agv_state['3F'], self.reservations_3f)
        
        # ParkingManager 功能已整合進 _find_smart_storage_spot
        
        self.qm_2f = PhysicalQueueManager(st_2f)
        self.qm_3f = PhysicalQueueManager(st_3f)
        
        self.zm_2f = ZoneManager(st_2f, capacity=4)
        self.zm_3f = ZoneManager(st_3f, capacity=4)
        
        self.wave_totals = {}
        for o in self.all_tasks_raw:
            wid = str(o.get('WAVE_ID', 'UNKNOWN')) 
            self.wave_totals[wid] = self.wave_totals.get(wid, 0) + 1
            
        self.monitor = LiveMonitor()

    def _init_spots(self, grid, floor):
        r, c = grid.shape
        for i in range(r):
            for j in range(c):
                if grid[i][j] == 1: self.valid_storage_spots[floor].add((i, j))
        
        for sid, info in self.shelf_coords.items():
            f = info['floor']
            p = info['pos']
            if f == floor and p in self.valid_storage_spots[floor]:
                self.shelf_occupancy[floor].add(p)
                if floor == '2F': self.pos_to_sid_2f[p] = sid
                else: self.pos_to_sid_3f[p] = sid

    def _load_inventory(self):
        path = os.path.join(BASE_DIR, 'data', 'master', 'item_inventory.csv')
        inv = defaultdict(list)
        try:
            df = pd.read_csv(path, dtype=str)
            cols = [c.upper() for c in df.columns]
            part_col = next((c for c in cols if 'PART' in c), None)
            cell_col = next((c for c in cols if 'CELL' in c or 'LOC' in c), None)
            if part_col and cell_col:
                for _, r in df.iterrows():
                    inv[str(r[part_col]).strip()].append(str(r[cell_col]).strip())
        except: pass
        return inv

    def _load_all_tasks(self):
        tasks = []
        path_out = os.path.join(BASE_DIR, 'data', 'transaction', 'wave_orders.csv')
        try:
            df_out = pd.read_csv(path_out)
            df_out.columns = [c.upper() for c in df_out.columns]
            date_col = next((c for c in df_out.columns if 'DATETIME' == c), None)
            if not date_col: date_col = next((c for c in df_out.columns if 'DATE' in c or 'TIME' in c), None)
            if date_col:
                df_out['datetime'] = pd.to_datetime(df_out[date_col])
                df_out = df_out.dropna(subset=['datetime'])
                if 'LOC' not in df_out.columns: df_out['LOC'] = ''
                tasks.extend(df_out.to_dict('records'))
        except: pass
        tasks.sort(key=lambda x: x['datetime'])
        return tasks 

    def _assign_locations_smartly(self, tasks):
        part_shelf_map = {}
        valid_shelves = list(self.shelf_coords.keys())
        for t in tasks:
            part = str(t.get('PARTNO', '')).strip()
            loc = str(t.get('LOC', '')).strip()
            if len(loc) >= 9:
                if part not in part_shelf_map: part_shelf_map[part] = loc 
        for t in tasks:
            loc = str(t.get('LOC', '')).strip()
            if len(loc) >= 9: continue 
            part = str(t.get('PARTNO', '')).strip()
            if part in part_shelf_map: t['LOC'] = part_shelf_map[part]
            else:
                cands = self.inventory_map.get(part, [])
                if cands:
                    t['LOC'] = cands[0]
                    part_shelf_map[part] = cands[0]
                elif valid_shelves:
                    chosen = f"{random.choice(valid_shelves)}-A-A01"
                    t['LOC'] = chosen
                    part_shelf_map[part] = chosen

    def _get_strict_spawn_spot(self, grid, used_spots, floor):
        rows, cols = grid.shape
        candidates = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 0: candidates.append((r,c))
        if not candidates:
             for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == 1: candidates.append((r,c))
        random.shuffle(candidates)
        return candidates[0] if candidates else (0,0)

    # [V75] 密度導向停車位搜尋 (Density-Aware Parking)
    def _find_smart_storage_spot(self, start_pos, valid_spots, occupied_spots, shelf_occupied_spots, agv_pool, grid, limit=50, avoid_high_density=False):
        if start_pos is None: return list(valid_spots)[:5]
        candidates = []
        
        # 建立簡易熱度圖
        heatmap = Counter()
        if avoid_high_density:
            for s in agv_pool.values():
                r, c = s['pos']
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        heatmap[(r+dr, c+dc)] += 1

        sample_spots = random.sample(list(valid_spots), min(limit*3, len(valid_spots)))
        
        for spot in sample_spots:
            if spot not in shelf_occupied_spots and spot not in occupied_spots:
                dist = abs(spot[0]-start_pos[0]) + abs(spot[1]-start_pos[1])
                
                # 密度懲罰
                density_penalty = 0
                if avoid_high_density:
                    density_penalty = heatmap[spot] * 20 # 附近每有一台車，成本激增
                
                obstacle_count = 0
                for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                    nr, nc = spot[0]+dr, spot[1]+dc
                    if not (0<=nr<grid.shape[0] and 0<=nc<grid.shape[1]): obstacle_count+=1
                    elif grid[nr][nc] == -1: obstacle_count+=1
                    elif (nr, nc) in shelf_occupied_spots: obstacle_count+=1
                
                island_penalty = 0
                if obstacle_count >= 3: island_penalty = 1000 
                
                total_score = dist + density_penalty + island_penalty + random.uniform(0, 10)
                candidates.append((total_score, spot))
        
        candidates.sort(key=lambda x: x[0])
        return [x[1] for x in candidates[:limit]]

    def _load_map_correct(self, filename, rows, cols):
        path = os.path.join(BASE_DIR, 'data', 'master', filename)
        if not os.path.exists(path): path = path.replace('.xlsx', '.csv')
        try:
            if filename.endswith('.xlsx'): df = pd.read_excel(path, header=None)
            else: df = pd.read_csv(path, header=None)
        except: return np.full((rows, cols), 0)
        raw_grid = df.iloc[0:rows, 0:cols].fillna(0).values 
        final_grid = np.full((rows, cols), -1.0) 
        r_in = min(raw_grid.shape[0], rows)
        c_in = min(raw_grid.shape[1], cols)
        final_grid[0:r_in, 0:c_in] = raw_grid[0:r_in, 0:c_in]
        return final_grid

    def _load_shelf_coords(self):
        path = os.path.join(BASE_DIR, 'data', 'mapping', 'shelf_coordinate_map.csv')
        coords = {}
        try:
            df = pd.read_csv(path)
            for _, r in df.iterrows():
                coords[str(r['shelf_id'])] = {'floor': r['floor'], 'pos': (int(r['y']), int(r['x']))}
        except: pass
        return coords

    def _init_stations(self):
        sts = {}
        def find_stations(grid):
            candidates = []
            rows, cols = grid.shape
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == 2: candidates.append((r, c))
            candidates.sort() 
            return candidates
        cands_2f = find_stations(self.grid_2f)
        for i, pos in enumerate(cands_2f): sts[f"2F_{i + 1}"] = {'floor': '2F', 'pos': pos}
        cands_3f = find_stations(self.grid_3f)
        for i, pos in enumerate(cands_3f): sts[f"3F_{i + 1}"] = {'floor': '3F', 'pos': pos}
        return sts

    def _is_physically_connected(self, grid, start, end):
        if grid[start[0]][start[1]] == -1 or grid[end[0]][end[1]] == -1: return False
        if abs(start[0] - end[0]) + abs(start[1] - end[1]) <= 1: return True
        q = deque([start])
        visited = {start}
        steps = 0
        while q:
            steps += 1
            if steps > 2000: return False 
            curr = q.popleft()
            if curr == end: return True
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = curr[0]+dr, curr[1]+dc
                if 0<=nr<grid.shape[0] and 0<=nc<grid.shape[1] and grid[nr][nc] != -1:
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        q.append((nr, nc))
        return False

    def write_move_events(self, writer, path, floor, agv_id, res_table, edge_res_table):
        if not path or len(path) < 2: return
        for i in range(len(path) - 1):
            curr_pos, curr_t = path[i]
            next_pos, next_t = path[i+1]
            
            # 1. 鎖點
            res_table[next_t].add(next_pos) 
            
            # 2. 鎖邊 (Edge Locking)
            # 記錄: 在 curr_t 時間點，從 curr_pos 移動到 next_pos
            edge_res_table[curr_t].add((curr_pos, next_pos))

            writer.writerow([
                self.to_dt(curr_t), self.to_dt(next_t), floor, f"AGV_{agv_id}",
                curr_pos[1], curr_pos[0], next_pos[1], next_pos[0], 'AGV_MOVE', ''
            ])
        if path:
            res_table[path[-1][1]].add(path[-1][0])

    def _cleanup_reservations(self, res_table, edge_res_table, current_time, last_clean_time):
        cutoff = current_time - 120
        if cutoff > last_clean_time + 200:
            for t in range(int(last_clean_time), int(cutoff)):
                res_table.pop(t, None) 
                edge_res_table.pop(t, None) # 同步清理邊緣鎖
            return cutoff 
        return last_clean_time

    def _clear_future_reservations(self, res_table, pos, start_t, duration=600):
        for t in range(int(start_t), int(start_t + duration)):
            if t in res_table and pos in res_table[t]:
                res_table[t].remove(pos)
                if not res_table[t]: del res_table[t]

    def _move_agv_segment(self, start_p, end_p, start_t, loaded, agv_name, floor, astar, traffic_ctrl, w_evt, res_table, edge_res_table, grid, is_returning=False, agv_pool=None, reason_label="GENERIC", hold_duration=0):
        curr = start_p
        target = end_p
        t = start_t
        start_wait = t
        TIMEOUT_LIMIT = 60 
        
        if not self._is_physically_connected(grid, curr, target):
            t += 120
            w_evt.writerow([self.to_dt(t-120), self.to_dt(t), floor, agv_name, curr[1], curr[0], target[1], target[0], 'AGV_MOVE', 'TELE_UNREACHABLE'])
            self.monitor.log_teleport(reason_label, 'Unreach')
            return target, t, True, None

        retry_count = 0 

        while curr != target:
            # 死鎖判定
            if t - start_wait > TIMEOUT_LIMIT:
                success, retreat_pos, retreat_time = traffic_ctrl.attempt_backtrack(curr, target, t, w_evt, floor, agv_name)
                if success:
                    curr = retreat_pos
                    t += retreat_time
                    start_wait = t 
                    continue
                
                t += 60 
                w_evt.writerow([self.to_dt(t-60), self.to_dt(t), floor, agv_name, curr[1], curr[0], target[1], target[0], 'AGV_MOVE', 'TELE_DEADLOCK'])
                self.monitor.log_teleport(reason_label, 'Stuck')
                return target, t, True, None

            # [V75] A* 搜尋 (包含方向性與邊緣鎖定)
            path, _ = astar.find_path(curr, target, t, is_loaded=loaded, ignore_dynamic=False)
            
            # [V75] 主動救援檢測 (Active Rescue Detection)
            # 如果找不到路，且已經嘗試了一段時間，檢查是否有靜態貨架擋路
            if not path and (t - start_wait > 5): 
                # 簡易檢測：如果目標點周圍被貨架包圍，或者路徑上必須穿過某個貨架
                # 這裡做一個簡單判定：如果目標就是被佔用的儲位 (且我不是要去放貨，而是要去穿過它)
                # 為了簡化，如果卡太久，我們回報 BLOCKED，讓外層去判斷需不需要救援
                
                # 檢查目標點是否是靜態障礙物
                if target in astar.shelf_occupancy:
                     return curr, t, False, {'type': 'BLOCKED', 'pos': target}
                
                # 檢查下一步是否是靜態障礙物 (簡單 Raycast)
                if curr[0]!=target[0] or curr[1]!=target[1]:
                    dr = 1 if target[0]>curr[0] else -1 if target[0]<curr[0] else 0
                    dc = 1 if target[1]>curr[1] else -1 if target[1]<curr[1] else 0
                    check_pos = (curr[0]+dr, curr[1]+dc)
                    if check_pos in astar.shelf_occupancy:
                        return curr, t, False, {'type': 'BLOCKED', 'pos': check_pos}

            if not path and (t - start_wait > 3): 
                success, penalty = traffic_ctrl.clear_path_obstacles(curr, target, t, w_evt, floor, agv_name)
                if success:
                    t += int(penalty)
                    continue

            if path:
                self.write_move_events(w_evt, path, floor, agv_name.replace("AGV_", ""), res_table, edge_res_table)
                arrival_t = path[-1][1]
                arrival_pos = path[-1][0]
                lock_time = hold_duration if hold_duration > 0 else 60
                
                for lock_t in range(arrival_t, arrival_t + lock_time):
                    res_table[lock_t].add(arrival_pos)

                t = arrival_t
                curr = target
                start_wait = t 
            else:
                backoff_time = min(2 ** retry_count, 5) 
                for k in range(backoff_time): res_table[t + k].add(curr)
                t += backoff_time
                retry_count += 1
                time.sleep(0.01) 
                    
        return curr, t, False, None

    def run(self):
        if not self.all_tasks_raw: return
        self.base_time = self.all_tasks_raw[0]['datetime']
        self.to_dt = lambda sec: self.base_time + timedelta(seconds=sec)
        
        # [V75] 初始化 A* (需傳入 edge_reservations)
        astar_2f = TimeAwareAStar(self.grid_2f, self.reservations_2f, self.edge_reservations_2f, self.shelf_occupancy['2F'])
        astar_3f = TimeAwareAStar(self.grid_3f, self.reservations_3f, self.edge_reservations_3f, self.shelf_occupancy['3F'])
        
        w_evt = BatchWriter(os.path.join(LOG_DIR, 'simulation_events.csv'), ['start_time', 'end_time', 'floor', 'obj_id', 'sx', 'sy', 'ex', 'ey', 'type', 'text'])
        f_kpi = open(os.path.join(LOG_DIR, 'simulation_kpi.csv'), 'w', newline='', encoding='utf-8')
        w_kpi = csv.writer(f_kpi)
        w_kpi.writerow(['finish_time', 'type', 'wave_id', 'is_delayed', 'date', 'workstation', 'total_in_wave', 'deadline_ts'])

        df_tasks = pd.DataFrame(self.all_tasks_raw)
        grouped_waves = df_tasks.groupby('WAVE_ID')
        
        task_queue_2f = deque()
        task_queue_3f = deque()
        
        for wave_id, wave_df in grouped_waves:
            wave_2f = wave_df[wave_df['LOC'].str.startswith('2')].copy()
            wave_3f = wave_df[wave_df['LOC'].str.startswith('3')].copy()
            task_queue_2f.extend(self.processor.process_wave(wave_2f, '2F'))
            task_queue_3f.extend(self.processor.process_wave(wave_3f, '3F'))
            
        total_tasks = len(task_queue_2f) + len(task_queue_3f)
        
        # 初始化輸出
        for floor in ['2F', '3F']:
            for sid, info in self.stations.items():
                if info['floor'] == floor:
                    w_evt.writerow([self.to_dt(0), self.to_dt(1), floor, f"WS_{sid}", info['pos'][1], info['pos'][0], info['pos'][1], info['pos'][0], 'STATION_STATUS', f'WHITE|IDLE|Waiting'])
            for agv_id, state in self.agv_state[floor].items():
                pos = state['pos']
                w_evt.writerow([self.to_dt(0), self.to_dt(1), floor, f"AGV_{agv_id}", pos[1], pos[0], pos[1], pos[0], 'AGV_MOVE', 'INIT'])

        done_count = 0
        queues = {'2F': task_queue_2f, '3F': task_queue_3f}
        astars = {'2F': astar_2f, '3F': astar_3f}
        q_mgrs = {'2F': self.qm_2f, '3F': self.qm_3f}
        z_mgrs = {'2F': self.zm_2f, '3F': self.zm_3f} 
        
        wait_counts = defaultdict(int) 

        for floor in ['2F', '3F']:
            queue = queues[floor]
            astar = astars[floor]
            agv_pool = self.agv_state[floor]
            res_table = self.reservations_2f if floor=='2F' else self.reservations_3f
            edge_res_table = self.edge_reservations_2f if floor=='2F' else self.edge_reservations_3f
            grid = self.grid_2f if floor=='2F' else self.grid_3f
            traffic = self.traffic_2f if floor=='2F' else self.traffic_3f
            q_mgr = q_mgrs[floor]
            z_mgr = z_mgrs[floor]
            
            # Helper maps for Rescue logic
            pos_to_sid = self.pos_to_sid_2f if floor=='2F' else self.pos_to_sid_3f
            shelf_occupancy = self.shelf_occupancy[floor]

            print(f"🏁 進入樓層 {floor} 迴圈，佇列長度: {len(queue)}")

            while queue:
                time.sleep(0.00001)

                best_agv = min(agv_pool, key=lambda k: agv_pool[k]['time'])
                agv_ready_time = agv_pool[best_agv]['time']
                agv_pos = agv_pool[best_agv]['pos']
                
                if grid[agv_pos[0]][agv_pos[1]] == -1: 
                    agv_pos = self._get_strict_spawn_spot(grid, set(), floor)
                    self.agv_state[floor][best_agv]['pos'] = agv_pos
                
                current_t = agv_ready_time
                
                # Task Fetching
                task = queue[0]
                
                # Check Priority Logic for Rescue Tasks
                is_rescue = task.get('priority', 0) > 100
                
                if not is_rescue:
                    task_dt = task.get('datetime')
                    if task_dt:
                        task_relative_sec = (task_dt - self.base_time).total_seconds()
                        if current_t < task_relative_sec:
                            current_t = int(task_relative_sec)

                target_st = task['stops'][-1]['station']
                
                # 如果是 Buffer (Rescue 任務)，不需要檢查 Zone/Queue
                if target_st != 'BUFFER':
                    if target_st not in q_mgr.station_queues:
                        queue.popleft(); continue

                    # Zone Control
                    if not z_mgr.can_enter(target_st):
                        wait_counts[best_agv] += 1
                        self.agv_state[floor][best_agv]['time'] += 5
                        continue
                    
                    z_mgr.enter(target_st)
                    wait_counts[best_agv] = 0

                    is_jammed = q_mgr.is_station_jammed(target_st, current_t)
                    target_pos, is_ready_to_work = q_mgr.get_target_for_agv(target_st, best_agv)
                    
                    if not target_pos or is_jammed:
                        z_mgr.exit(target_st) 
                        self.agv_state[floor][best_agv]['time'] += 5
                        continue
                else:
                    target_pos = task['stops'][-1]['pos']

                task = queue.popleft() 
                shelf_id = task['shelf_id']
                
                # 取得貨架當前位置 (可能已被移庫)
                shelf_pos = self.shelf_coords[shelf_id]['pos'] if shelf_id in self.shelf_coords else None
                if not shelf_pos: 
                    if target_st != 'BUFFER': z_mgr.exit(target_st)
                    done_count += 1; continue
                
                # 1. Load Shelf
                agv_pos, current_t, tele_1, err_info_1 = self._move_agv_segment(
                    agv_pos, shelf_pos, current_t, False, f"AGV_{best_agv}", 
                    floor, astar, traffic, w_evt, res_table, edge_res_table, grid, reason_label="LOAD"
                )
                
                # [V75] 主動救援檢測：如果 Load 失敗是因為被擋住
                if err_info_1 and err_info_1['type'] == 'BLOCKED':
                    blk_pos = err_info_1['pos']
                    blk_sid = pos_to_sid.get(blk_pos, "Unknown")
                    print(f"🚨 [Rescue] {best_agv} 被 {blk_sid} 擋住，發起救援任務！")
                    
                    # 1. 找個遠離熱區的空位
                    safe_spot = self._find_smart_storage_spot(blk_pos, self.valid_storage_spots[floor], 
                        {s['pos'] for k,s in agv_pool.items()}, shelf_occupancy, agv_pool, grid, limit=10, avoid_high_density=True)[0]
                    
                    # 2. 建立救援訂單
                    rescue_task = {
                        'wave_id': 'RESCUE', 'shelf_id': blk_sid, 'priority': 999,
                        'stops': [{'station': 'BUFFER', 'pos': safe_spot, 'time': 0}],
                        'raw_orders': [], 'datetime': None
                    }
                    queue.appendleft(task) # 把原本的任務放回去 (還沒做完)
                    queue.appendleft(rescue_task) # 插隊救援任務
                    
                    # 讓車子暫時退避
                    if target_st != 'BUFFER': z_mgr.exit(target_st)
                    self.agv_state[floor][best_agv]['time'] += 10
                    continue

                if tele_1: self.monitor.log_success('Load')
                
                w_evt.writerow([self.to_dt(current_t), self.to_dt(current_t+5), floor, f"AGV_{best_agv}", shelf_pos[1], shelf_pos[0], shelf_pos[1], shelf_pos[0], 'SHELF_LOAD', f"Task_{done_count}"])
                current_t += 5
                if shelf_pos in shelf_occupancy: shelf_occupancy.remove(shelf_pos)
                current_shelf_pos = shelf_pos

                # --- 2. Visit Station / Buffer ---
                if target_st == 'BUFFER':
                    # 救援任務邏輯：直接把貨搬去 Buffer
                    agv_pos, current_t, tele_2, _ = self._move_agv_segment(
                        current_shelf_pos, target_pos, current_t, True, f"AGV_{best_agv}",
                        floor, astar, traffic, w_evt, res_table, edge_res_table, grid, reason_label="RESCUE_MOVE"
                    )
                    # 搬到了，更新貨架位置
                    if shelf_id in self.shelf_coords: self.shelf_coords[shelf_id]['pos'] = agv_pos
                    shelf_occupancy.add(agv_pos)
                    pos_to_sid[agv_pos] = shelf_id
                    w_evt.writerow([self.to_dt(current_t), self.to_dt(current_t+5), floor, f"AGV_{best_agv}", agv_pos[1], agv_pos[0], agv_pos[1], agv_pos[0], 'SHELF_UNLOAD', f"Rescued {shelf_id}"])
                    current_t += 5
                    self.monitor.log_success('Rescue')
                    self.agv_state[floor][best_agv]['pos'] = agv_pos
                    self.agv_state[floor][best_agv]['time'] = current_t
                    continue # 救援結束，直接進下個循環

                else:
                    # 正常工作站邏輯
                    while True:
                        # Heartbeat
                        for t_lock in range(current_t, current_t + 10): res_table[t_lock].add(current_shelf_pos)

                        next_q_pos, is_processing = q_mgr.get_target_for_agv(target_st, best_agv)
                        if not next_q_pos:
                            current_t += 2; continue
                        
                        current_step_hold_time = 120 
                        if is_processing:
                            for stop in task['stops']:
                                if stop['station'] == target_st:
                                    current_step_hold_time = stop['time'] + 30; break

                        new_pos, new_t, tele_2, _ = self._move_agv_segment(
                            current_shelf_pos, next_q_pos, current_t, True, f"AGV_{best_agv}",
                            floor, astar, traffic, w_evt, res_table, edge_res_table, grid, 
                            reason_label="QUEUE", hold_duration=current_step_hold_time
                        )
                        
                        if new_pos == next_q_pos:
                            q_mgr.update_position(target_st, best_agv, new_pos)
                            current_shelf_pos = new_pos
                            current_t = new_t
                        else:
                            current_t = new_t
                        
                        if is_processing and current_shelf_pos == next_q_pos: break             

                    if tele_2: self.monitor.log_success('Visit')
                    q_mgr.set_processing_time(target_st, current_t)
                    stop_time = task['stops'][0]['time'] 
                    leave_t = current_t + stop_time
                    
                    for t_proc in range(current_t, int(leave_t) + 30): res_table[t_proc].add(current_shelf_pos)
                    
                    w_evt.writerow([self.to_dt(current_t), self.to_dt(leave_t), floor, f"WS_{target_st}", current_shelf_pos[1], current_shelf_pos[0], current_shelf_pos[1], current_shelf_pos[0], 'PICKING', f"Processing"])
                    current_t = int(leave_t)
                    
                    # Exit Logic
                    exit_pos = q_mgr.get_exit_spot(target_st)
                    if exit_pos:
                        while True:
                            new_pos, new_t, _, _ = self._move_agv_segment(
                                current_shelf_pos, exit_pos, current_t, True, f"AGV_{best_agv}",
                                floor, astar, traffic, w_evt, res_table, edge_res_table, grid, 
                                reason_label="EXIT", hold_duration=0
                            )
                            if new_pos == exit_pos:
                                current_shelf_pos = new_pos; current_t = new_t; break
                            else:
                                current_t = new_t; time.sleep(0.01)

                    q_mgr.release_station(target_st, best_agv)
                    z_mgr.exit(target_st)
                    
                    st_proc_pos = q_mgr.station_queues[target_st]['slots'][0] 
                    real_proc_pos = (st_proc_pos[0], 1)
                    self._clear_future_reservations(res_table, real_proc_pos, current_t - 10, duration=600)

                    # 3. Return (Smart Density Parking)
                    # [V75] 使用密度感知尋找還貨點
                    candidates = self._find_smart_storage_spot(
                        current_shelf_pos, self.valid_storage_spots[floor], 
                        {s['pos'] for k,s in agv_pool.items()}, shelf_occupancy, agv_pool, grid, limit=20, avoid_high_density=True
                    )
                    drop_pos = candidates[0] if candidates else shelf_pos # Fallback to original
                    if grid[drop_pos[0]][drop_pos[1]] == -1: drop_pos = self._get_strict_spawn_spot(grid, set(), floor)

                    current_shelf_pos, current_t, tele_3, _ = self._move_agv_segment(
                        current_shelf_pos, drop_pos, current_t, True, f"AGV_{best_agv}",
                        floor, astar, traffic, w_evt, res_table, edge_res_table, grid,
                        is_returning=True, agv_pool=agv_pool, reason_label="RETURN", hold_duration=120
                    )
                    if tele_3: self.monitor.log_success('Return')

                    w_evt.writerow([self.to_dt(current_t), self.to_dt(current_t+5), floor, f"AGV_{best_agv}", drop_pos[1], drop_pos[0], drop_pos[1], drop_pos[0], 'SHELF_UNLOAD', 'Done'])
                    current_t += 5
                    
                    shelf_occupancy.add(drop_pos)
                    self.shelf_coords[shelf_id]['pos'] = drop_pos
                    if floor == '2F': pos_to_sid[drop_pos] = shelf_id
                    
                    self.agv_state[floor][best_agv]['pos'] = drop_pos
                    self.agv_state[floor][best_agv]['time'] = current_t
                        
                    # 4. Park
                    candidates = self._find_smart_storage_spot(drop_pos, self.valid_storage_spots[floor], {s['pos'] for k,s in agv_pool.items()}, shelf_occupancy, agv_pool, grid, limit=5, avoid_high_density=True)
                    park_spot = candidates[0] if candidates else None

                    if park_spot:
                        current_shelf_pos, current_t, tele_4, _ = self._move_agv_segment(
                            drop_pos, park_spot, current_t, False, f"AGV_{best_agv}",
                            floor, astar, traffic, w_evt, res_table, edge_res_table, grid, 
                            is_returning=False, agv_pool=agv_pool, reason_label="PARK_FINAL", hold_duration=120
                        )
                        if not tele_4:
                            self.agv_state[floor][best_agv]['pos'] = park_spot
                            self.agv_state[floor][best_agv]['time'] = current_t
                            w_evt.writerow([self.to_dt(current_t), self.to_dt(current_t+1), floor, f"AGV_{best_agv}", park_spot[1], park_spot[0], park_spot[1], park_spot[0], 'PARKING', 'Hidden'])
                            self.monitor.log_success('Park')

                    for raw_o in task['raw_orders']:
                        wid = raw_o.get('WAVE_ID', 'UNK')
                        ttype = 'INBOUND' if 'RECEIVING' in wid else 'OUTBOUND'
                        total_wave_count = self.wave_totals.get(wid, 0)
                        deadline_dt = self.to_dt(0) + timedelta(hours=4)
                        st_label = f"WS_{task['stops'][-1]['station']}" 
                        w_kpi.writerow([self.to_dt(current_t), ttype, wid, 'N', self.to_dt(current_t).date(), st_label, total_wave_count, deadline_dt])

                done_count += 1
                if done_count % 10 == 0:
                     print(f"✅ Progress: {done_count}/{total_tasks} completed.")
                
                if done_count % 50 == 0 or done_count == total_tasks: 
                    self.last_clean_time_2f = self._cleanup_reservations(self.reservations_2f, self.edge_reservations_2f, current_t, self.last_clean_time_2f)
                    self.last_clean_time_3f = self._cleanup_reservations(self.reservations_3f, self.edge_reservations_3f, current_t, self.last_clean_time_3f)
                    self.monitor.print_status(done_count, total_tasks, agv_pool, len([t for t in queue if t.get('priority',0)>100]))

        w_evt.close()
        f_kpi.close()
        print(f"\n✅ 模擬完成！ Stats: {self.monitor.stats}")

if __name__ == "__main__":
    AdvancedSimulationRunner().run()