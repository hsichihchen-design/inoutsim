import pandas as pd
import numpy as np
import os
import sys
import csv
import time
from datetime import datetime, timedelta

# 引入引擎
from engine.configs import SimConfig
from engine.physics import MapWorld, AGV

# ==========================================
# 設定檔案路徑
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_TRX_DIR = os.path.join(BASE_DIR, 'data', 'transaction')
DATA_MAP_DIR = os.path.join(BASE_DIR, 'data', 'mapping')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

os.makedirs(LOG_DIR, exist_ok=True)

WAVE_FILE = 'wave_orders.csv'
SHELF_MAP_FILE = 'shelf_coordinate_map.csv'

# 輸出檔名
TRACE_FILE = 'simulation_trace.csv'  # 視覺化用 (肥大)
KPI_FILE = 'simulation_kpi.csv'      # 報表用 (輕量)

class SimulationRunner:
    def __init__(self, enable_trace=False, speed_up=True):
        print(f"🚀 [Step 3] 啟動高效能模擬引擎 (Trace={enable_trace}, TimeSkip={speed_up})...")
        
        self.enable_trace = enable_trace
        self.speed_up = speed_up
        
        # 1. 載入設定
        self.cfg = SimConfig(BASE_DIR)
        
        # 2. 初始化世界
        print("🌍 初始化物理世界...")
        self.world_2f = MapWorld('2F', '2F_map.xlsx', BASE_DIR)
        self.world_3f = MapWorld('3F', '3F_map.xlsx', BASE_DIR)
        self.shelf_map = self._load_shelf_map()
        
        # 3. 初始化 AGV
        agv_count_2f = self.cfg.get('planned_staff_2f', 5)
        agv_count_3f = self.cfg.get('planned_staff_3f', 5)
        
        self.agvs_2f = self._init_agvs(self.world_2f, agv_count_2f, start_id=1)
        self.agvs_3f = self._init_agvs(self.world_3f, agv_count_3f, start_id=101)
        self.all_agvs = self.agvs_2f + self.agvs_3f
        
        print(f"🤖 AGV 就位: 2F({len(self.agvs_2f)}台), 3F({len(self.agvs_3f)}台)")

        # 4. 載入訂單
        print("📦 載入訂單波次...")
        self.orders = self._load_orders()
        self.order_queue = [] 
        
        # 統計變數
        self.stats = {
            'total_orders': len(self.orders),
            'completed': 0,
            'delayed': 0,
            'total_travel_dist': 0
        }

    def _load_shelf_map(self):
        path = os.path.join(DATA_MAP_DIR, SHELF_MAP_FILE)
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到座標表: {path}")
        df = pd.read_csv(path)
        mapping = {}
        for _, row in df.iterrows():
            s_id = str(row['shelf_id'])
            if s_id not in mapping:
                mapping[s_id] = {'floor': row['floor'], 'pos': (int(row['x']), int(row['y']))}
        return mapping

    def _init_agvs(self, world, count, start_id):
        agvs = []
        candidates = world.charging_stations + world.workstations
        # 補空地
        if len(candidates) < count:
            rows, cols = world.grid.shape
            for r in range(rows):
                for c in range(cols):
                    if world.grid[r][c] == 0: candidates.append((r,c))
        
        for i in range(count):
            pos = candidates[i % len(candidates)]
            is_under = (world.grid[pos[0]][pos[1]] == 1)
            agv = AGV(start_id + i, world.floor, pos, is_under_shelf=is_under)
            agvs.append(agv)
        return agvs

    def _load_orders(self):
        path = os.path.join(DATA_TRX_DIR, WAVE_FILE)
        if not os.path.exists(path):
            print("⚠️ 找不到波次訂單，請先執行 Step 2")
            return []
        df = pd.read_csv(path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['WAVE_DEADLINE'] = pd.to_datetime(df['WAVE_DEADLINE'])
        df = df.sort_values('datetime')
        return df.to_dict('records')

    def run(self, duration_days=1):
        # 計算總秒數
        duration_seconds = duration_days * 24 * 3600
        print(f"\n🎬 開始模擬 (模擬天數: {duration_days} 天, 總秒數: {duration_seconds})...")
        
        # 準備 Log 檔案
        f_trace = None
        trace_writer = None
        if self.enable_trace:
            f_trace = open(os.path.join(LOG_DIR, TRACE_FILE), 'w', newline='', encoding='utf-8')
            trace_writer = csv.writer(f_trace)
            trace_writer.writerow(['timestamp', 'floor', 'agv_id', 'x', 'y', 'status', 'battery'])

        f_kpi = open(os.path.join(LOG_DIR, KPI_FILE), 'w', newline='', encoding='utf-8')
        kpi_writer = csv.writer(f_kpi)
        kpi_writer.writerow(['order_id', 'create_time', 'start_time', 'finish_time', 'wave_deadline', 'is_delayed', 'processing_time'])

        # 初始化時鐘
        if not self.orders:
            print("❌ 沒有訂單可模擬")
            return
            
        sim_start_time = self.orders[0]['datetime']
        current_time = sim_start_time
        end_time = sim_start_time + timedelta(seconds=duration_seconds)
        
        tick = 0
        real_start = time.time()
        
        try:
            while current_time < end_time:
                
                # --- [優化 1] 時空跳躍邏輯 ---
                # 條件：沒有待辦任務 AND 所有車都閒置 AND 還有未來訂單
                all_idle = all(agv.status == 'IDLE' for agv in self.all_agvs)
                if self.speed_up and not self.order_queue and all_idle and self.orders:
                    next_order_time = self.orders[0]['datetime']
                    time_diff = (next_order_time - current_time).total_seconds()
                    
                    if time_diff > 5: # 如果空檔超過 5 秒才跳
                        # print(f"⏩ [Time Skip] 跳過閒置 {int(time_diff)} 秒 (至 {next_order_time})")
                        current_time = next_order_time
                        # 這裡不增加 tick 計數，因為那是邏輯 tick，我們只在乎時間推進
                        continue

                # 1. 釋放訂單
                while self.orders and self.orders[0]['datetime'] <= current_time:
                    order = self.orders.pop(0)
                    
                    # [模擬簡化] 每 N 張單合併成一個搬運任務 (減少計算量)
                    # 假設這是 "Task Generator" 的工作
                    # 這裡為了演示，隨機抽樣 10% 的單產生搬運需求
                    if tick % 10 == 0: 
                        target_shelf_id = list(self.shelf_map.keys())[tick % len(self.shelf_map)]
                        target_info = self.shelf_map[target_shelf_id]
                        
                        task = {
                            'order_obj': order, # 保留原始訂單資訊以便記錄 KPI
                            'shelf_id': target_shelf_id,
                            'floor': target_info['floor'],
                            'target_pos': target_info['pos'],
                            'start_time': current_time
                        }
                        self.order_queue.append(task)
                    
                    # 如果該單沒有產生搬運任務 (譬如被合併了)，直接視為完成
                    else:
                        self.stats['completed'] += 1
                        # 這種「虛擬完成」的單也要記 KPI 嗎？視需求，這裡先略過

                # 2. 調度邏輯 (Dispatcher)
                # (跟之前一樣，為省篇幅省略註解)
                for task in list(self.order_queue):
                    assigned = False
                    if task['floor'] == '2F':
                        agvs = self.agvs_2f
                        world = self.world_2f
                    else:
                        agvs = self.agvs_3f
                        world = self.world_3f
                    
                    # 簡單派車：找最近閒置
                    best_agv = None
                    min_dist = 9999
                    for agv in agvs:
                        if agv.status == 'IDLE' and not agv.current_task:
                            dist = abs(agv.pos[0] - task['target_pos'][0]) + abs(agv.pos[1] - task['target_pos'][1])
                            if dist < min_dist:
                                min_dist = dist
                                best_agv = agv
                    
                    if best_agv:
                        success = best_agv.plan_path(world, task['target_pos'])
                        if success:
                            best_agv.status = 'MOVING'
                            best_agv.current_task = task
                            self.order_queue.remove(task)
                            assigned = True

                # 3. 物理更新
                for floor, agvs, world in [('2F', self.agvs_2f, self.world_2f), ('3F', self.agvs_3f, self.world_3f)]:
                    other_positions = {a.pos for a in agvs}
                    for agv in agvs:
                        if agv.status == 'MOVING':
                            others = other_positions - {agv.pos}
                            moved, new_pos = agv.move_step(world, others)
                            
                            # 到達檢查
                            if agv.pos == agv.current_task['target_pos']:
                                # 完成任務，記錄 KPI
                                task = agv.current_task
                                order = task['order_obj']
                                finish_time = current_time
                                duration = (finish_time - task['start_time']).total_seconds()
                                is_delayed = finish_time > order['WAVE_DEADLINE']
                                
                                # 寫入 KPI CSV
                                kpi_writer.writerow([
                                    f"ORD_{tick}", 
                                    order['datetime'], 
                                    task['start_time'], 
                                    finish_time, 
                                    order['WAVE_DEADLINE'], 
                                    'Y' if is_delayed else 'N', 
                                    duration
                                ])
                                
                                agv.status = 'IDLE'
                                agv.current_task = None
                                self.stats['completed'] += 1
                                if is_delayed: self.stats['delayed'] += 1
                        
                        # --- [優化 2] 選擇性 Log ---
                        # 只有在 enable_trace 開啟時才寫入座標
                        if self.enable_trace and trace_writer:
                            trace_writer.writerow([
                                current_time, floor, agv.id, 
                                agv.pos[1], agv.pos[0], 
                                agv.status, f"{agv.battery:.1f}"
                            ])

                # 4. 時間推進
                current_time += timedelta(seconds=1)
                tick += 1
                
                # 顯示進度 (每 1000 tick 更新一次，避免拖慢速度)
                if tick % 1000 == 0:
                    elapsed = time.time() - real_start
                    # 預估剩餘時間
                    progress = (current_time - sim_start_time).total_seconds() / duration_seconds
                    eta = elapsed / progress * (1 - progress) if progress > 0 else 0
                    print(f"\r⏳ 進度: {progress*100:.1f}% | 時間: {current_time} | 完成: {self.stats['completed']} | 延遲: {self.stats['delayed']} | ETA: {eta/60:.1f}分", end='')

        except KeyboardInterrupt:
            print("\n🛑 使用者中斷")
        finally:
            if f_trace: f_trace.close()
            f_kpi.close()
            print(f"\n✅ 模擬結束！")
            print(f"   -> 視覺化軌跡: {'已儲存' if self.enable_trace else '未啟用 (simulation_trace.csv)'}")
            print(f"   -> 績效報表: 已儲存 (simulation_kpi.csv)")
            print(f"   -> 總完成任務: {self.stats['completed']}")

if __name__ == "__main__":
    # --- 使用者設定區 ---
    
    # 模式 A: 視覺化除錯 (跑 1 小時，開啟 Trace，不跳躍時間)
    # runner = SimulationRunner(enable_trace=True, speed_up=False)
    # runner.run(duration_days=0.04) # 約 1 小時
    
    # 模式 B: 長期績效模擬 (跑 7 天，關閉 Trace，開啟跳躍)
    runner = SimulationRunner(enable_trace=False, speed_up=True)
    runner.run(duration_days=7) # 試跑一週