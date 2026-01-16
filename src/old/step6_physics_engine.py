import pandas as pd
import numpy as np
import os
import heapq
import csv
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
DATA_MAP_DIR = os.path.join(BASE_DIR, 'data', 'master')
DATA_TRX_DIR = os.path.join(BASE_DIR, 'data', 'transaction')

# ==========================================
# A* Pathfinding (核心尋路演算法)
# ==========================================
def heuristic(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])

def find_path(grid, start, goal):
    rows, cols = grid.shape
    if not (0 <= start[0] < rows and 0 <= start[1] < cols): return None
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols): return None
    
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    steps = 0
    max_steps = 5000 

    while open_set:
        steps += 1
        if steps > max_steps: return None 

        current = heapq.heappop(open_set)[1]
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (current[0] + dr, current[1] + dc)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor[0]][neighbor[1]] == 1 and neighbor != goal: continue
                
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None

# ==========================================
# Simulation Classes (物理實體)
# ==========================================
class TrafficManager:
    def __init__(self): self.occupied = {}
    def request(self, agv_id, floor, pos):
        key = (floor, pos[0], pos[1])
        if key in self.occupied and self.occupied[key] != agv_id:
            return False
        return True
    def update(self, agv_id, floor, old, new):
        old_k = (floor, old[0], old[1])
        if old_k in self.occupied and self.occupied[old_k] == agv_id: 
            del self.occupied[old_k]
        self.occupied[(floor, new[0], new[1])] = agv_id

class AGV:
    def __init__(self, uid, floor, start_pos):
        self.id = uid
        self.floor = floor
        self.pos = start_pos
        self.path = []
        self.state = 'IDLE' 
        self.task_data = None 

    def assign_task(self, grid, target, order_data):
        path = find_path(grid, self.pos, target)
        if path:
            self.path = path
            self.state = 'MOVING'
            self.task_data = order_data
            return True
        return False

    def step(self, tm):
        if self.state == 'MOVING' and self.path:
            next_pos = self.path[0]
            if tm.request(self.id, self.floor, next_pos):
                tm.update(self.id, self.floor, self.pos, next_pos)
                self.pos = next_pos
                self.path.pop(0)
                if not self.path:
                    self.state = 'WORKING' 
                    return 'ARRIVED', self.pos
                return 'MOVED', self.pos
            else:
                return 'BLOCKED', self.pos
        elif self.state == 'WORKING':
            self.state = 'IDLE'
            done_task = self.task_data
            self.task_data = None
            return 'DONE', done_task
        return 'IDLE', None

class PhysicsSim:
    def __init__(self):
        print("🚀 [Step 6] 物理模擬引擎啟動 (Strict Sort Mode)...")
        self.tm = TrafficManager()
        self.w2 = self._load_grid('2F_map.xlsx')
        self.w3 = self._load_grid('3F_map.xlsx')
        self.agvs = []
        self._spawn_agvs(self.w2, '2F', 1, 8)
        self._spawn_agvs(self.w3, '3F', 101, 8)
        self.orders = self._load_orders()
        self.shelf_map = self._load_shelf_map()

    def _load_grid(self, f):
        p = os.path.join(DATA_MAP_DIR, f)
        if not os.path.exists(p): p = p.replace('.xlsx', '.csv')
        try: return pd.read_excel(p, header=None).fillna(0).values
        except: 
            try: return pd.read_csv(p, header=None).fillna(0).values
            except: return np.zeros((30,60))

    def _load_shelf_map(self):
        p = os.path.join(BASE_DIR, 'data', 'mapping', 'shelf_coordinate_map.csv')
        try: 
            df = pd.read_csv(p)
            return {str(r['shelf_id']): {'floor': r['floor'], 'pos': (int(r['y']), int(r['x']))} for _, r in df.iterrows()}
        except: return {}

    def _spawn_agvs(self, grid, floor, start, count):
        rows, cols = grid.shape
        cands = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] in [0,3]]
        if not cands: cands = [(0,0)]
        import random
        random.shuffle(cands)
        for i in range(count):
            pos = cands[i%len(cands)]
            a = AGV(start+i, floor, pos)
            self.agvs.append(a)
            self.tm.update(a.id, floor, pos, pos)

    def _load_orders(self):
        p = os.path.join(DATA_TRX_DIR, 'wave_orders.csv')
        try:
            try: df = pd.read_csv(p, encoding='utf-8-sig')
            except: df = pd.read_csv(p, encoding='cp950')
            
            # [關鍵修正] 強制重新構建 datetime 並排序
            # 有時候 CSV 存檔後順序會亂，或者格式被誤判
            if 'DATE' in df.columns and 'TIME' in df.columns:
                df['datetime'] = pd.to_datetime(df['DATE'].astype(str) + ' ' + df['TIME'].astype(str))
            else:
                df['datetime'] = pd.to_datetime(df['datetime'])
            
            # 這裡進行嚴格排序！
            df = df.sort_values('datetime', ascending=True).reset_index(drop=True)
            
            if 'WAVE_DEADLINE' in df.columns:
                df['WAVE_DEADLINE'] = pd.to_datetime(df['WAVE_DEADLINE'], errors='coerce')
                
            print(f"📦 訂單載入完成: {len(df)} 筆")
            print(f"   📅 時間範圍: {df['datetime'].min()} ~ {df['datetime'].max()}")
            return df.to_dict('records')
        except Exception as e: 
            print(f"❌ 讀取訂單失敗: {e}")
            return []

    def run(self, max_ticks=86400): # 預設跑 24 小時 (1天)
        # Setup Logs
        f_evt = open(os.path.join(LOG_DIR, 'simulation_events.csv'), 'w', newline='', encoding='utf-8')
        w_evt = csv.writer(f_evt)
        w_evt.writerow(['start_time', 'end_time', 'floor', 'obj_id', 'sx', 'sy', 'ex', 'ey', 'type', 'text'])
        
        f_kpi = open(os.path.join(LOG_DIR, 'simulation_kpi.csv'), 'w', newline='', encoding='utf-8')
        w_kpi = csv.writer(f_kpi)
        w_kpi.writerow(['finish_time', 'type', 'wave_id', 'is_delayed', 'date', 'workstation'])

        if not self.orders: 
            print("⚠️ 無訂單資料")
            return
            
        sim_time = self.orders[0]['datetime']
        oidx = 0
        completed_count = 0
        
        print(f"🎬 開始模擬... (起始時間: {sim_time})")
        
        for tick in range(max_ticks):
            sim_time += timedelta(seconds=1)
            
            # [心跳包] 每 10 分鐘寫入一筆，確保 Visualizer 時間軸能撐開
            if tick % 600 == 0 and self.agvs:
                a0 = self.agvs[0]
                w_evt.writerow([sim_time, sim_time, a0.floor, f"AGV_{a0.id}", a0.pos[1], a0.pos[0], a0.pos[1], a0.pos[0], 'HEARTBEAT', ''])

            # 1. Dispatch Orders
            while oidx < len(self.orders) and self.orders[oidx]['datetime'] <= sim_time:
                ord_data = self.orders[oidx]
                
                target_pos = None
                target_floor = '2F'
                
                if self.shelf_map:
                    import random
                    sid = random.choice(list(self.shelf_map.keys()))
                    info = self.shelf_map[sid]
                    target_pos = info['pos']
                    target_floor = info['floor']
                else:
                    target_pos = (10, 10)
                
                candidates = [a for a in self.agvs if a.state=='IDLE' and a.floor==target_floor]
                if candidates:
                    candidates.sort(key=lambda a: abs(a.pos[0]-target_pos[0]) + abs(a.pos[1]-target_pos[1]))
                    assigned = candidates[0].assign_task(self.w2 if target_floor=='2F' else self.w3, target_pos, ord_data)
                    if assigned:
                        oidx += 1 
                    else:
                        break 
                else:
                    break 
                
            # 2. Physics Step
            for agv in self.agvs:
                status, res = agv.step(self.tm)
                
                if status == 'MOVED':
                    w_evt.writerow([
                        sim_time-timedelta(seconds=1), sim_time, agv.floor, f"AGV_{agv.id}",
                        res[1], res[0], res[1], res[0], 'AGV_MOVE', '' 
                    ])
                elif status == 'DONE':
                    completed_count += 1
                    task = res
                    deadline = task.get('WAVE_DEADLINE')
                    if pd.isna(deadline): deadline = sim_time + timedelta(hours=1)
                    is_delayed = 'Y' if sim_time > deadline else 'N'
                    
                    w_kpi.writerow([
                        sim_time, 'PICKING', task.get('WAVE_ID', 'W_DEFAULT'), 
                        is_delayed, sim_time.date(), 'WS_1'
                    ])

            if tick % 5000 == 0: 
                pct = (tick / max_ticks) * 100
                print(f"\r⏳ 進度: {pct:.1f}% | 時間: {sim_time} | 完成單量: {completed_count}", end='')

        f_evt.close()
        f_kpi.close()
        print(f"\n✅ 模擬結束！共完成 {completed_count} 張訂單")

if __name__ == "__main__":
    # 如果要跑 24 小時，請設 86400
    # 如果要跑 10 天，請設 864000
    PhysicsSim().run(max_ticks=86400)