import pandas as pd
import numpy as np
import os
import sys
import json
import random
import heapq
from datetime import datetime, timedelta
from collections import defaultdict, deque

# 路徑設定
CURRENT_FILE_PATH = os.path.abspath(__file__)
SRC_DIR = os.path.dirname(CURRENT_FILE_PATH)
BASE_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path: sys.path.insert(0, SRC_DIR)

import step0_config as config
from step3_strategies import StrategyEngine

DATA_TRX_DIR = os.path.join(BASE_DIR, 'data', 'transaction')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# ==========================================
# 輔助類別 (Helpers)
# ==========================================
class Task:
    def __init__(self, row_data, task_type):
        self.id = str(random.randint(100000, 999999))
        self.type = task_type 
        self.data = row_data
        self.part_id = str(row_data.get('PARTNO', '')).strip()
        self.cust_id = str(row_data.get('PARTCUSTID', 'MISC')).strip().upper()
        
        # 識別 Follow 細分類 (TC1, TC2, KH...)
        self.follow_type = str(row_data.get('FOLLOW_TYPE', 'OTHER')) if task_type == 'FOLLOW' else None
        
        # 數量解析
        try: self.qty = int(float(str(row_data.get('QTY', '1'))))
        except: self.qty = 1 
        
        # 時間計算參數
        self.is_repack = str(row_data.get('REPACK', '0')).strip() == '1'
        self.pick_seconds = config.CONFIG['PICK_TIME_REPACK'] if self.is_repack else config.CONFIG['PICK_TIME_NORMAL']

        self.wave_id = row_data.get('WAVE_ID', 'NA')
        self.deadline = None
        if 'DEADLINE' in row_data and pd.notna(row_data['DEADLINE']):
            self.deadline = pd.to_datetime(row_data['DEADLINE'])
            
        self.release_time = pd.to_datetime(row_data.get('DATETIME')) if 'DATETIME' in row_data else None

class Station:
    def __init__(self, st_id, floor):
        self.id = st_id
        self.floor = floor 
        self.state = 'IDLE' 
        self.current_task_group = None 
        self.current_shelf = None
        self.remaining_time = 0 
        self.last_task_type = None 
        self.assigned_wave_id = None
        self.assigned_points = set()

# ==========================================
# 波次規劃器 (WavePlanner)
# ==========================================
class WavePlanner:
    def __init__(self, max_stations_per_floor=8):
        self.max_st = max_stations_per_floor

    def plan_wave(self, wave_id, tasks, start_time, floor):
        if not tasks: return 0, {}
        
        deadline = tasks[0].deadline
        if not deadline or deadline <= start_time:
            available_seconds = 3600
        else:
            available_seconds = (deadline - start_time).total_seconds()
            if available_seconds < 600: available_seconds = 600

        point_load = defaultdict(float)
        for t in tasks:
            point_load[t.cust_id] += t.pick_seconds

        total_workload = sum(point_load.values())
        needed_stations = int(np.ceil(total_workload / available_seconds))
        needed_stations = max(1, min(needed_stations, self.max_st))
        
        sorted_points = sorted(point_load.items(), key=lambda x: x[1], reverse=True)
        station_slots = [(0, i) for i in range(needed_stations)]
        heapq.heapify(station_slots)
        
        assignment_map = defaultdict(set)
        for pid, load in sorted_points:
            current_load, st_idx = heapq.heappop(station_slots)
            assignment_map[st_idx].add(pid)
            heapq.heappush(station_slots, (current_load + load, st_idx))
            
        return needed_stations, assignment_map

# ==========================================
# 模擬引擎 (Engine)
# ==========================================
class SimulationEngine:
    def __init__(self):
        print("🚀 [Engine] 初始化模擬引擎 (修正版)...")
        self.strategy = StrategyEngine()
        self.planner = WavePlanner(max_stations_per_floor=8)
        
        self.queues = {
            'URGENT': deque(), 
            'FOLLOW': deque(), 
            'INBOUND': deque(), 
            'REPLENISHMENT': deque()
        }
        self.wave_queues = defaultdict(deque) 
        self.wave_totals = {}
        
        self.stats_total = {
            'INBOUND': 0, 'REPLENISHMENT': 0, 'TC1': 0, 'FOLLOW_OTHER': 0
        }
        
        self.active_waves = {} 
        self.finished_waves = set()
        self.stations = self._init_stations()
        self.shelf_locks = {}
        
        self.kpi = {'shipped': 0, 'received': 0, 'stockouts': 0, 'overflows': 0}
        self._load_tasks_and_set_time()

    def _init_stations(self):
        sts = []
        for i in range(8): sts.append(Station(f"2F_ST_{i+1}", '2F'))
        for i in range(8): sts.append(Station(f"3F_ST_{i+1}", '3F'))
        return sts

    def _load_tasks_and_set_time(self):
            print("   -> 載入任務 (啟動嚴格庫存過濾)...")
            files = {'STANDARD': 'tasks_standard.csv', 'URGENT': 'tasks_urgent.csv',
                    'INBOUND': 'tasks_inbound.csv', 'FOLLOW': 'tasks_follow.csv',
                    'REPLENISHMENT': 'tasks_replenishment.csv'}
            
            all_timestamps = []
            skipped_ghost_parts = 0 # [NEW] 統計被踢除的幽靈訂單
            
            for t_type, fname in files.items():
                path = os.path.join(DATA_TRX_DIR, fname)
                if not os.path.exists(path): continue
                try: df = pd.read_csv(path, dtype=str, encoding='utf-8')
                except: df = pd.read_csv(path, dtype=str, encoding='cp950')
                df.columns = [c.upper().strip() for c in df.columns]
                
                if 'DATETIME' in df.columns:
                    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
                    df = df.sort_values('DATETIME')
                    all_timestamps.extend(df['DATETIME'].dropna().tolist())

                for _, row in df.iterrows():
                    # [MODIFIED] 配合新資料結構，只讀取 PARTNO (移除 FRCD)
                    pid = str(row.get('PARTNO', '')).strip()

                    # [CRITICAL FILTER] 嚴格過濾：需確認料號在 Map 中，且「實際持有可揀貨的儲位」
                    if t_type != 'INBOUND':
                        # 1. 基本檢查：是否在樓層對照表中
                        is_in_map = pid in self.strategy.part_floor_map
                        # 2. 實體檢查：是否有可用的儲位庫存 (排除暫存區)
                        # 注意：依賴 Step 3 的 inventory 結構，若無儲位 key 不會存在或為空
                        has_physical_stock = False
                        if is_in_map:
                             if pid in self.strategy.inventory and self.strategy.inventory[pid]:
                                 has_physical_stock = True
                        
                        if not has_physical_stock:
                            skipped_ghost_parts += 1
                            # 除錯用：若想看是哪些料號被踢除，可取消下行註解
                            # if skipped_ghost_parts <= 5: print(f"   [Filter] Skip {pid}: No physical stock.")
                            continue

                    task = Task(row, t_type)
                    # 重新確保 Task 內部的 ID 也是對的
                    task.part_id = pid 

                    # 放入對應佇列
                    if t_type == 'STANDARD':
                        self.wave_queues[task.wave_id].append(task)
                        self.wave_totals[task.wave_id] = self.wave_totals.get(task.wave_id, 0) + 1
                    elif t_type == 'INBOUND':
                        self.queues['INBOUND'].append(task)
                        self.stats_total['INBOUND'] += 1
                    elif t_type == 'REPLENISHMENT':
                        self.queues['REPLENISHMENT'].append(task)
                        self.stats_total['REPLENISHMENT'] += 1
                    elif t_type == 'FOLLOW':
                        self.queues['FOLLOW'].append(task)
                        if task.follow_type == 'TC1': self.stats_total['TC1'] += 1
                        else: self.stats_total['FOLLOW_OTHER'] += 1
                    elif t_type == 'URGENT':
                        self.queues['URGENT'].append(task)

            print(f"   🚫 已自動過濾 {skipped_ghost_parts} 筆無效庫存訂單 (Ghost Orders)")

            # 設定時間
            sim_start_str = config.CONFIG.get("SIMULATION_START_TIME", "09:00")
            if all_timestamps:
                min_data_time = min(all_timestamps)
                max_data_time = max(all_timestamps)
                base_date = min_data_time.date()
                work_start_time = datetime.strptime(f"{base_date} {sim_start_str}", "%Y-%m-%d %H:%M")
                
                self.current_time = max(min_data_time, work_start_time)
                self.end_time = max_data_time + timedelta(hours=2)
                print(f"🗓️ 模擬範圍: {self.current_time} ~ {self.end_time}")
            else:
                self.current_time = datetime.now()
                self.end_time = self.current_time + timedelta(hours=1)

    def run(self):
        print(f"▶️ 模擬開始: {self.current_time} ~ {self.end_time}")
        delta = timedelta(seconds=1) 
        try:
            while self.current_time <= self.end_time:
                self._check_wave_start()
                self._dispatch_tasks()
                self._update_stations()
                
                # [MODIFIED] 每 30 分鐘 Print 一次完整戰情板
                if self.current_time.second == 0 and self.current_time.minute % 30 == 0:
                    self._print_dashboard() # 改名為 dashboard
                    
                self.current_time += delta
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted")
        print("\n⏹ Done.")

    def _diagnose_stuck_wave(self):
        if not self.active_waves: return
        wave_id = list(self.active_waves.keys())[0]
        queue = self.wave_queues[wave_id]
        if not queue: return 
        print(f"\n🔍 [診斷] 波次 {wave_id} 剩餘 {len(queue)} 筆任務卡關中...")
        stuck_cust_ids = defaultdict(int)
        for t in queue: stuck_cust_ids[t.cust_id] += 1
        sorted_stuck = sorted(stuck_cust_ids.items(), key=lambda x:x[1], reverse=True)[:5]
        print(f"   📉 卡關任務據點分佈: {sorted_stuck}")
        print("   ------------------------------------------------")

    def _check_wave_start(self):
        # [EDF] Earliest Deadline First
        candidates = []
        for wave_id, queue in self.wave_queues.items():
            if not queue or wave_id in self.finished_waves: continue
            first_task = queue[0]
            if self.current_time >= first_task.release_time:
                try:
                    if first_task.deadline: dl = first_task.deadline
                    else:
                        time_part = wave_id.split('_')[-1]
                        date_part = wave_id.split('_')[-2]
                        dl = datetime.strptime(f"{date_part} {time_part}", "%Y%m%d_%H%M")
                    candidates.append((wave_id, queue, dl))
                except:
                    candidates.append((wave_id, queue, datetime.max))
        
        if not candidates: return
        candidates.sort(key=lambda x: x[2])
        
        active_count = len(self.active_waves)
        MAX_CONCURRENT_WAVES = 2
        
        for wave_id, queue, dl in candidates:
            if wave_id in self.active_waves: continue
            if active_count < MAX_CONCURRENT_WAVES:
                self._activate_wave(wave_id, queue)
                active_count += 1
                print(f"   🚀 [EDF啟動] {self.current_time.strftime('%H:%M')} | {wave_id}")
            else:
                break

    def _activate_wave(self, wave_id, tasks_queue):
        tasks_2f = [t for t in tasks_queue if self.strategy.part_floor_map.get(t.part_id) == '2F']
        tasks_3f = [t for t in tasks_queue if self.strategy.part_floor_map.get(t.part_id) == '3F']
        
        plan_2f = self.planner.plan_wave(wave_id, tasks_2f, self.current_time, '2F')
        plan_3f = self.planner.plan_wave(wave_id, tasks_3f, self.current_time, '3F')
        
        self._assign_stations_to_wave(wave_id, '2F', plan_2f)
        self._assign_stations_to_wave(wave_id, '3F', plan_3f)
        self.active_waves[wave_id] = {'start_time': self.current_time}

    def _assign_stations_to_wave(self, wave_id, floor, plan):
        needed_count, assign_map = plan
        if needed_count == 0: return
        target_stations = [s for s in self.stations if s.floor == floor]
        for idx, point_set in assign_map.items():
            if idx < len(target_stations):
                st = target_stations[idx]
                st.assigned_wave_id = wave_id
                st.assigned_points = point_set

    def _dispatch_tasks(self):
        idle_stations = [s for s in self.stations if s.state == 'IDLE']
        random.shuffle(idle_stations)
        
        for st in idle_stations:
            selected = None
            # P1: Wave
            if st.assigned_wave_id and st.assigned_wave_id in self.wave_queues:
                wave_id = st.assigned_wave_id
                queue = self.wave_queues[wave_id]
                if queue:
                    my_tasks = [t for t in queue if t.cust_id in st.assigned_points]
                    if my_tasks: selected = self._try_batch_standard_for_station(st, wave_id, my_tasks[0])
                    else:
                        st.assigned_wave_id = None
                        st.assigned_points = set()
                else:
                    self.finished_waves.add(wave_id)
                    if wave_id in self.active_waves: del self.active_waves[wave_id]
                    st.assigned_wave_id = None

            # P2~P5
            if not selected and not st.assigned_wave_id:
                if self.queues['URGENT'] and self.current_time >= self.queues['URGENT'][0].release_time:
                    selected = self._try_batch_general(st, self.queues['URGENT'], 'URGENT')
                elif self.queues['FOLLOW']:
                    selected = self._try_batch_general(st, self.queues['FOLLOW'], 'FOLLOW')
                elif self.queues['INBOUND'] and self.current_time >= self.queues['INBOUND'][0].release_time:
                    selected = self._try_process_inbound(st)
                elif self.queues['REPLENISHMENT'] and self.current_time >= self.queues['REPLENISHMENT'][0].release_time:
                    selected = self._try_batch_general(st, self.queues['REPLENISHMENT'], 'REPLENISHMENT')

            if selected: self._assign_job_to_station(st, selected)

    def _try_batch_standard_for_station(self, station, wave_id, first_task):
        st_demand = {first_task.part_id: first_task.qty}
        sug = self.strategy.find_stock_for_outbound(st_demand, st_demand)
        main_queue = self.wave_queues[wave_id]
        if not sug: return None 
        target = sug[0]
        if target in self.shelf_locks: return None
        batch = []
        rem = deque()
        for t in main_queue:
            if t.part_id == first_task.part_id and t.cust_id in station.assigned_points: batch.append(t)
            else: rem.append(t)
        self.wave_queues[wave_id] = rem
        self.shelf_locks[target] = station.id
        return {'type': 'STANDARD', 'shelf_id': target, 'tasks': batch, 'mode': 'OUTBOUND'}

    def _try_batch_general(self, station, queue, t_type):
        task = queue[0]
        if self.strategy.part_floor_map.get(task.part_id, '2F') != station.floor: return None
        st_demand = {task.part_id: task.qty}
        sug = self.strategy.find_stock_for_outbound(st_demand, st_demand)
        if not sug:
            queue.popleft()
            return None
        target = sug[0]
        if target in self.shelf_locks: return None
        task = queue.popleft()
        self.shelf_locks[target] = station.id
        return {'type': t_type, 'shelf_id': target, 'tasks': [task], 'mode': 'OUTBOUND'}

    def _try_process_inbound(self, station):
        task = self.queues['INBOUND'][0]
        if self.strategy.part_floor_map.get(task.part_id, '2F') != station.floor: return None
        plan, status = self.strategy.find_best_bin_for_inbound(task.data.get('FRCD'), task.data.get('PARTNO'), task.qty)
        if not plan:
            self.queues['INBOUND'].popleft()
            self.kpi['overflows'] += 1
            return None
        sid, bid = plan
        if sid in self.shelf_locks: return None
        self.queues['INBOUND'].popleft()
        self.shelf_locks[sid] = station.id
        return {'type': 'INBOUND', 'shelf_id': sid, 'tasks': [task], 'target_bin': bid, 'mode': 'INBOUND'}

    def _assign_job_to_station(self, station, job_pack):
        if station.last_task_type == job_pack['type']:
            move_time = config.sample_time(config.CONFIG["TIME_SHELF_SWITCH_SAME_MODE"])
        else:
            move_time = config.sample_time(config.CONFIG["TIME_MODE_SWITCH_ARRIVAL"])
        if job_pack['mode'] == 'OUTBOUND': op_time = sum(t.pick_seconds for t in job_pack['tasks'])
        else: op_time = config.sample_time(config.CONFIG["TIME_PUTAWAY_PER_BIN"])
        station.state = 'WORKING'
        station.current_task_group = job_pack
        station.current_shelf = job_pack['shelf_id']
        station.remaining_time = move_time + op_time
        station.last_task_type = job_pack['type']

    def _update_stations(self):
        for st in self.stations:
            if st.state == 'WORKING':
                st.remaining_time -= 1
                if st.remaining_time <= 0: self._complete_job(st)
                    
    def _complete_job(self, station):
        job = station.current_task_group
        if station.current_shelf in self.shelf_locks: del self.shelf_locks[station.current_shelf]
        if job['mode'] == 'OUTBOUND': self.kpi['shipped'] += len(job['tasks'])
        else: self.kpi['received'] += len(job['tasks'])
        station.state = 'IDLE'
        station.current_task_group = None
        station.current_shelf = None

    # ==========================================
    # 戰情儀表板 (Dashboard 2.0)
    # ==========================================
    def _print_dashboard(self):
        # 清除上一行的殘影 (視情況)
        print("\n" + "="*80)
        curr_str = self.current_time.strftime("%Y-%m-%d %H:%M")
        print(f"🕒 時間: {curr_str} | 🏭 資源負載: 2F [{self._get_st_load('2F')}/8] | 3F [{self._get_st_load('3F')}/8]")
        
        # --- 1. 任務佇列概況 ---
        # 計算完成率
        def fmt_prog(key):
            total = self.stats_total.get(key, 0)
            if total == 0: return "0/0"
            done = total - len(self.queues[key])
            # TC1 和 Follow 要額外扣除 active 但還沒做完的? 暫時忽略，以 queue 為準
            if key == 'FOLLOW': # 特殊處理 Follow 拆分
                rem_tc1 = sum(1 for t in self.queues['FOLLOW'] if t.follow_type == 'TC1')
                rem_oth = len(self.queues['FOLLOW']) - rem_tc1
                done_tc1 = self.stats_total['TC1'] - rem_tc1
                done_oth = self.stats_total['FOLLOW_OTHER'] - rem_oth
                return f"TC1:{done_tc1}/{self.stats_total['TC1']} | Oth:{done_oth}/{self.stats_total['FOLLOW_OTHER']}"
            return f"{done}/{total}"

        print(f"📋 任務進度: Urg: {len(self.queues['URGENT'])} | Inb: {fmt_prog('INBOUND')} | Rep: {fmt_prog('REPLENISHMENT')} | {fmt_prog('FOLLOW')}")
        
        # --- 2. 波次詳細追蹤 ---
        print(f"🌊 波次狀態追蹤 (今日):")
        # 收集所有已出現過的波次 (Active + Finished)
        all_waves = set(self.active_waves.keys()) | self.finished_waves
        # 過濾出今天的波次
        today_waves = [w for w in all_waves if w.split('_')[1] == self.current_time.strftime("%Y%m%d")]
        today_waves.sort()
        
        if not today_waves:
            print("   (尚無波次啟動)")
        
        for wid in today_waves:
            # 狀態判斷
            status = ""
            total = self.wave_totals.get(wid, 1)
            
            if wid in self.finished_waves:
                status = "✅ 完成"
                # 如果能記錄完成時間更好，這裡暫時顯示 Finished
            elif wid in self.active_waves:
                remain = len(self.wave_queues[wid])
                pct = int(((total - remain) / total) * 100)
                status = f"🟡 進行中 ({pct}%)"
            else:
                status = "⚪ 等待中"

            # 檢查是否 Delay
            # 從 ID 解析 Deadline: W_20250701_0925
            try:
                time_str = wid.split('_')[-1]
                dl_dt = datetime.strptime(f"{self.current_time.date()} {time_str}", "%Y-%m-%d %H%M")
                
                delay_str = ""
                if self.current_time > dl_dt and wid not in self.finished_waves:
                    diff = (self.current_time - dl_dt).total_seconds() / 60
                    status = f"🔴 延遲! ({int(diff)} min)"
                elif wid in self.finished_waves:
                    status += " (準時)" # 簡化，實際要記完成時間才能算準時
            except:
                dl_dt = "N/A"

            print(f"   - {wid}: {status} | 總單數: {total}")

        # --- 3. 閒置原因診斷 (若有空站) ---
        idle_2f = [s for s in self.stations if s.floor == '2F' and s.state == 'IDLE']
        idle_3f = [s for s in self.stations if s.floor == '3F' and s.state == 'IDLE']
        
        if idle_2f or idle_3f:
            print(f"⚠️ 閒置診斷: 2F有{len(idle_2f)}站閒置, 3F有{len(idle_3f)}站閒置")
            # 簡單分析：是因為沒任務? 還是被鎖定?
            # 檢查是否有任何佇列有東西
            has_task = any(len(q) > 0 for q in self.queues.values()) or any(len(q) > 0 for q in self.wave_queues.values())
            if not has_task:
                print("   -> 原因: 全場無任務 (訂單已清空)")
            else:
                print("   -> 原因: 任務存在但無法指派 (可能是缺料、料架鎖定或樓層不符)")

        print("="*80 + "\n")

    def _get_st_load(self, floor):
        return sum(1 for s in self.stations if s.floor == floor and s.state == 'WORKING')

if __name__ == "__main__":
    engine = SimulationEngine()
    engine.run()