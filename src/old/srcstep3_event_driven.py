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
KPI_FILE = 'simulation_kpi.csv'

class EventSimulationRunner:
    def __init__(self):
        print(f"🚀 [Step 3.5] 啟動極速事件驅動引擎 (Event-Driven)...")
        
        self.cfg = SimConfig(BASE_DIR)
        
        # 1. 初始化世界
        print("🌍 初始化地圖...")
        self.world_2f = MapWorld('2F', '2F_map.xlsx', BASE_DIR)
        self.world_3f = MapWorld('3F', '3F_map.xlsx', BASE_DIR)
        self.shelf_map = self._load_shelf_map()
        
        # 2. 初始化 AGV
        agv_count_2f = self.cfg.get('planned_staff_2f', 8)
        agv_count_3f = self.cfg.get('planned_staff_3f', 8)
        
        self.agvs_2f = self._init_agvs(self.world_2f, agv_count_2f, start_id=1)
        self.agvs_3f = self._init_agvs(self.world_3f, agv_count_3f, start_id=101)
        self.all_agvs = self.agvs_2f + self.agvs_3f
        
        # --- [關鍵] AGV 狀態管理 ---
        # 記錄每台車何時會「解鎖」 (完成當前任務的時間)
        # 格式: { agv_id: unlock_datetime }
        self.agv_unlock_times = {} 
        
        # 3. 載入訂單
        print("📦 載入訂單...")
        self.orders = self._load_orders()
        self.order_queue = [] 
        
        # 統計
        self.stats = {'completed': 0, 'delayed': 0}

    def _load_shelf_map(self):
        path = os.path.join(DATA_MAP_DIR, SHELF_MAP_FILE)
        df = pd.read_csv(path)
        mapping = {}
        for _, row in df.iterrows():
            mapping[str(row['shelf_id'])] = {'floor': row['floor'], 'pos': (int(row['x']), int(row['y']))}
        return mapping

    def _init_agvs(self, world, count, start_id):
        agvs = []
        candidates = world.charging_stations + world.workstations
        if len(candidates) < count:
            rows, cols = world.grid.shape
            for r in range(rows):
                for c in range(cols):
                    if world.grid[r][c] == 0: candidates.append((r,c))
        
        for i in range(count):
            pos = candidates[i % len(candidates)]
            agv = AGV(start_id + i, world.floor, pos)
            agvs.append(agv)
        return agvs

    def _load_orders(self):
        path = os.path.join(DATA_TRX_DIR, WAVE_FILE)
        df = pd.read_csv(path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['WAVE_DEADLINE'] = pd.to_datetime(df['WAVE_DEADLINE'])
        df = df.sort_values('datetime')
        return df.to_dict('records')

    def get_travel_time(self, agv, target_pos):
        """
        [極速計算] 直接用曼哈頓距離 / 速度
        假設 AGV 速度 = 1 格/秒 (可調整)
        """
        dist = abs(agv.pos[0] - target_pos[0]) + abs(agv.pos[1] - target_pos[1])
        speed = 1.5 # 假設每秒跑 1.5 格
        seconds = dist / speed
        
        # 加上一點隨機變異 (模擬轉彎、減速)
        noise = 0 
        return timedelta(seconds=(seconds + noise))

    def run(self, duration_days=7):
        print(f"\n🎬 開始極速模擬 (天數: {duration_days})...")
        
        f_kpi = open(os.path.join(LOG_DIR, KPI_FILE), 'w', newline='', encoding='utf-8')
        kpi_writer = csv.writer(f_kpi)
        kpi_writer.writerow(['order_id', 'create_time', 'start_time', 'finish_time', 'wave_deadline', 'is_delayed', 'duration'])

        if not self.orders: return

        # 時間初始化
        sim_start_time = self.orders[0]['datetime']
        current_time = sim_start_time
        end_time = sim_start_time + timedelta(days=duration_days)
        
        # 雙指針優化訂單讀取
        order_idx = 0
        total_orders = len(self.orders)
        
        real_start = time.time()
        
        # ==========================================
        # [核心] 事件驅動迴圈
        # ==========================================
        while current_time < end_time and (order_idx < total_orders or self.order_queue or self.agv_unlock_times):
            
            # 1. [事件：釋放訂單]
            # 把所有「現在時間點之前」的訂單放入 Queue
            while order_idx < total_orders and self.orders[order_idx]['datetime'] <= current_time:
                order = self.orders[order_idx]
                
                # 這裡模擬：每張單都需要搬運 (或者您可以保留 % 10 的邏輯)
                # 為了壓力測試，我們假設每張單都是一個 Task
                target_shelf_id = list(self.shelf_map.keys())[order_idx % len(self.shelf_map)]
                target_info = self.shelf_map[target_shelf_id]
                
                task = {
                    'order_obj': order,
                    'floor': target_info['floor'],
                    'target_pos': target_info['pos'],
                    'start_time': current_time # 進入 Queue 的時間
                }
                self.order_queue.append(task)
                order_idx += 1

            # 2. [事件：檢查 AGV 完成]
            # 檢查有哪些 AGV 在這個時間點「解鎖」了 (任務完成)
            finished_agvs = []
            for agv_id, unlock_time in list(self.agv_unlock_times.items()):
                if unlock_time <= current_time:
                    # 任務完成!
                    del self.agv_unlock_times[agv_id] # 移除鎖定
                    
                    # 找回 AGV 物件更新狀態
                    agv = next((a for a in self.all_agvs if a.id == agv_id), None)
                    if agv:
                        agv.status = 'IDLE'
                        # 更新位置 (瞬移到目的地)
                        if agv.current_task:
                            agv.pos = agv.current_task['target_pos']
                            
                            # --- 記錄 KPI ---
                            task = agv.current_task
                            finish_time = unlock_time # 完成時間 = 解鎖時間
                            order = task['order_obj']
                            duration = (finish_time - task['start_time']).total_seconds()
                            is_delayed = finish_time > order['WAVE_DEADLINE']
                            
                            kpi_writer.writerow([
                                f"ORD_{order_idx}", order['datetime'], task['start_time'], 
                                finish_time, order['WAVE_DEADLINE'], 
                                'Y' if is_delayed else 'N', duration
                            ])
                            
                            self.stats['completed'] += 1
                            if is_delayed: self.stats['delayed'] += 1
                            
                            agv.current_task = None
            
            # 3. [事件：指派任務]
            # 嘗試把 Queue 裡的任務派給 IDLE 的車
            # 先將 Queue 分樓層
            # 簡單優化：只遍歷一次
            remaining_queue = []
            for task in self.order_queue:
                assigned = False
                
                # 根據樓層選車隊
                target_agvs = self.agvs_2f if task['floor'] == '2F' else self.agvs_3f
                
                # 找閒置車 (不在 unlock_times 裡的車就是 IDLE)
                available_agvs = [a for a in target_agvs if a.id not in self.agv_unlock_times]
                
                if available_agvs:
                    # 簡單策略：選第一台 (因為現在是算產能，不需算精確距離)
                    best_agv = available_agvs[0]
                    
                    # 計算耗時 (DES 核心)
                    travel_duration = self.get_travel_time(best_agv, task['target_pos'])
                    # 加上揀貨時間 (來自參數，例如 20秒)
                    pick_duration = timedelta(seconds=20) 
                    
                    total_duration = travel_duration + pick_duration
                    
                    # 鎖定 AGV
                    finish_time = current_time + total_duration
                    self.agv_unlock_times[best_agv.id] = finish_time
                    best_agv.current_task = task
                    best_agv.status = 'BUSY'
                    
                    assigned = True
                
                if not assigned:
                    remaining_queue.append(task)
            
            self.order_queue = remaining_queue

            # 4. [核心] 時間跳躍 (Time Warp)
            # 下一個關鍵時間點 = min(下一張訂單進來的時間, 最快一台車完成的時間)
            
            next_event_times = []
            
            # A. 下一張訂單時間
            if order_idx < total_orders:
                next_event_times.append(self.orders[order_idx]['datetime'])
            
            # B. 最快完成的車
            if self.agv_unlock_times:
                next_event_times.append(min(self.agv_unlock_times.values()))
            
            if next_event_times:
                next_wake_up = min(next_event_times)
                # 如果下個事件在未來，就跳過去；如果在過去(或現在)，就只加一點點時間避免無窮迴圈
                if next_wake_up > current_time:
                    current_time = next_wake_up
                else:
                    # 防止死鎖：如果時間沒推進，強制微調 1 秒
                    current_time += timedelta(seconds=1)
            else:
                # 沒有任何未來事件了 (訂單發完、車都做完)
                break
                
            # 顯示進度
            if self.stats['completed'] % 1000 == 0:
                print(f"\r🚀 時間: {current_time} | 完成: {self.stats['completed']} | 延遲: {self.stats['delayed']} | Queue: {len(self.order_queue)}", end='')

        f_kpi.close()
        print(f"\n\n✅ 模擬結束！耗時: {time.time() - real_start:.2f} 秒")
        print(f"   -> 總訂單: {total_orders}")
        print(f"   -> 完成數: {self.stats['completed']}")

if __name__ == "__main__":
    sim = EventSimulationRunner()
    sim.run(duration_days=180) # 直接挑戰半年！