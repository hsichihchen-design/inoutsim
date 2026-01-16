import pandas as pd
import numpy as np
import os
import pickle
from collections import defaultdict, deque

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
EVENTS_FILE = os.path.join(LOG_DIR, 'simulation_events.csv')
DATA_FILE = os.path.join(BASE_DIR, 'processed_sim_data.pkl')
# ----------------------------------------

def get_station_zones(grid, stations, capacity=4):
    """
    重新計算每個工作站的物理區域 (Center + Slots)
    這段邏輯必須與 simulation_core 中的 PhysicalZoneManager._init_slots 一致
    """
    rows, cols = grid.shape
    zones = {}
    
    QUEUE_MARKER = 4
    
    for sid, info in stations.items():
        center_pos = info['pos']
        valid_slots = []
        found_marker_slots = []
        max_search_dist = 10 
        
        # BFS 找排隊點
        q = deque([center_pos])
        visited = {center_pos}
        
        while q:
            curr = q.popleft()
            r, c = curr
            dist = abs(r - center_pos[0]) + abs(c - center_pos[1])
            if dist > max_search_dist: continue
            
            if grid[r][c] == QUEUE_MARKER and curr != center_pos:
                found_marker_slots.append(curr)
                
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    if grid[nr][nc] != -1: 
                         visited.add((nr, nc))
                         q.append((nr, nc))
        
        if found_marker_slots:
            found_marker_slots.sort(key=lambda p: abs(p[0]-center_pos[0]) + abs(p[1]-center_pos[1]))
            valid_slots = found_marker_slots[:capacity]
        else:
            # Fallback 邏輯
            q_backup = deque([center_pos])
            visited_backup = {center_pos}
            while q_backup and len(valid_slots) < capacity:
                curr = q_backup.popleft()
                if curr != center_pos and grid[curr[0]][curr[1]] != -1:
                    valid_slots.append(curr)
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nr, nc = curr[0]+dr, curr[1]+dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited_backup:
                        visited_backup.add((nr, nc))
                        q_backup.append((nr, nc))
        
        # 定義區域: Center 是工作點, Slots 是排隊點
        zones[sid] = {
            'center': center_pos,
            'slots': set(valid_slots),
            'all': set(valid_slots) | {center_pos}
        }
    return zones

def main():
    print("⚖️ [Step 9] 啟動嚴格物理驗證器 (Strict Validator)...")
    
    if not os.path.exists(EVENTS_FILE):
        print("❌ 找不到 Log 檔。")
        return

    # 1. 載入地圖與工作站資訊
    with open(DATA_FILE, 'rb') as f:
        sim_data = pickle.load(f)
    
    grid_2f = sim_data['grid_2f']
    grid_3f = sim_data['grid_3f']
    stations = sim_data['stations']
    
    # 2. 建立驗證區域 (Zones)
    stations_2f = {k:v for k,v in stations.items() if v['floor']=='2F'}
    stations_3f = {k:v for k,v in stations.items() if v['floor']=='3F'}
    
    zones = {}
    zones.update(get_station_zones(grid_2f, stations_2f))
    zones.update(get_station_zones(grid_3f, stations_3f))
    
    print(f"   已建立 {len(zones)} 個工作站的物理圍欄。")

    # 3. 讀取與處理 Events
    df = pd.read_csv(EVENTS_FILE)
    df['start_ts'] = pd.to_datetime(df['start_time'])
    df['end_ts'] = pd.to_datetime(df['end_time'])
    base_time = df['start_ts'].min()
    
    # 轉成相對秒數
    df['s_sec'] = (df['start_ts'] - base_time).dt.total_seconds().astype(int)
    df['e_sec'] = (df['end_ts'] - base_time).dt.total_seconds().astype(int)
    
    max_sim_time = df['e_sec'].max()
    
    # 4. 重播模擬 (Replay)
    # 我們需要知道每一秒，每一台 AGV 在哪裡
    # 為了效能，我們用事件驅動更新，而不是每秒掃描
    
    agv_positions = {} # {agv_id: (floor, r, c)}
    violations_work = [] # 違反 "工作點只能有1台"
    violations_zone = [] # 違反 "區域總數 < 5"
    
    # 依時間排序事件
    events = df.sort_values('s_sec').to_dict('records')
    
    # 建立時間軸檢查點 (每秒檢查一次最準確，但如果太慢可以改 5秒)
    check_interval = 1 
    current_event_idx = 0
    total_events = len(events)
    
    print(f"   開始重播 {max_sim_time} 秒的模擬歷史...")
    
    # 統計用
    station_stats = defaultdict(lambda: {'max_working': 0, 'max_total': 0, 'violation_sec': 0})
    
    for t in range(0, int(max_sim_time) + 1, check_interval):
        if t % 1000 == 0: print(f"   ⏳ Time: {t}s ...")
        
        # 更新 AGV 位置 (處理在這個時間點之前發生的所有移動)
        # 注意：我們只關心 AGV "靜止" 或 "佔用" 的位置。
        # 如果 AGV 正在移動中 (s_sec < t < e_sec)，它算在哪？
        # 嚴格來說，移動中佔用的是路徑。但為了簡化驗證工作站佔用，
        # 我們假設：如果 t >= e_sec，它到達了終點。如果 t < e_sec，它還在起點或路上。
        # 最嚴格的檢查是看 "到達後" 的停留狀態。
        
        while current_event_idx < total_events and events[current_event_idx]['s_sec'] <= t:
            e = events[current_event_idx]
            # 當事件開始時，我們雖然還沒到終點，但為了追蹤位置，我們先記錄它是 "Active"
            # 但真正的位置更新發生在 "到達" (end_time)
            # 不過，如果我們只更新 end_time，那移動中間會變成 "瞬移"。
            # 這裡採用：讀取該 AGV 在此時刻的最新已知位置。
            
            # 簡單做法：我們只看該 AGV "最新完成" 的位置
            # 或者更精確：看這個時間點，哪一個 Event 涵蓋了它
            current_event_idx += 1
            
        # 為了精確，我們不依賴 cursor，而是直接查詢每個 AGV 在時間 t 的狀態
        # 但那樣太慢。改用「狀態機」：
        # 依序讀取事件，維護 `current_positions`
    
    # === 優化版重播邏輯 ===
    # 我們改用 "區間樹" 概念的簡化版：
    # 每個 Station 在時間軸上都有計數器。
    # 遍歷所有 Events，如果是 "移動到工作站區域"，就在該時段 +1
    
    print("   正在構建工作站佔用時間軸 (Timeline Analysis)...")
    
    # station_occupancy[sid][time] = { 'working': count, 'queue': count }
    # 使用稀疏矩陣或字典紀錄變化點，避免記憶體爆掉
    # 但為了簡單，我們先用 NumPy Array (如果時間不長)
    
    timeline_len = int(max_sim_time) + 100
    # 記憶體優化：只存有問題的站
    # 我們直接針對 Event 進行判定
    
    # counters[sid][t] = count
    # 為了省記憶體，我們用 dict of dict，只存非零值? 不，用 numpy int8 應該夠 (時間 x 站點數)
    # 假設 20 個站 x 5000 秒 = 100,000，很小。
    
    station_ids = list(zones.keys())
    s_map = {sid: i for i, sid in enumerate(station_ids)}
    
    # shape: (num_stations, timeline_len)
    working_counts = np.zeros((len(station_ids), timeline_len), dtype=np.int8)
    total_counts = np.zeros((len(station_ids), timeline_len), dtype=np.int8)
    
    for _, row in df.iterrows():
        agv_id = row['obj_id']
        if not str(agv_id).startswith('AGV'): continue
        
        floor = row['floor']
        dest = (int(row['ey']), int(row['ex'])) # (row, col)
        
        # 這裡的邏輯：
        # 當 AGV 移動到 dest，並且停在那裡直到下一次移動開始
        # 這段時間 [end_sec, next_start_sec] 它是佔用 dest 的。
        # 我們需要找出這台 AGV 的 "下一次移動開始時間"
        pass 

    # 重新整理數據：依 AGV 分組，算出每個 AGV 的停留區間
    agv_groups = df.groupby('obj_id')
    
    for agv_id, group in agv_groups:
        if not str(agv_id).startswith('AGV'): continue
        
        group = group.sort_values('s_sec')
        records = group.to_dict('records')
        
        for i in range(len(records)):
            curr_e = records[i]
            floor = curr_e['floor']
            # 目的地座標 (row, col)
            r, c = int(curr_e['ey']), int(curr_e['ex']) 
            pos = (r, c)
            
            # 停留開始時間 = 移動結束時間
            stay_start = curr_e['e_sec']
            
            # 停留結束時間 = 下一個事件的開始時間 (如果沒有下一個，就假設停到最後)
            if i + 1 < len(records):
                stay_end = records[i+1]['s_sec']
            else:
                stay_end = int(max_sim_time)
            
            if stay_end <= stay_start: continue
            
            # 檢查這個位置是否屬於某個工作站
            # 這是效能瓶頸，我們要快速反查
            # 建立反查表 (在 loop 外做一次)
            
            # ... (下面會移到 loop 外) ...
            
            # 標記時間軸
            # 為了效能，這裡只標記 "與工作站有關" 的位置
            # 使用我們預先建立的反查表
            pass

    # === 真正的執行邏輯 ===
    
    # 1. 建立座標反查表 (Coord -> Station ID & Type)
    coord_map = {} # (floor, r, c) -> (sid, type='center'|'slot')
    for sid, z in zones.items():
        floor = stations[sid]['floor']
        # Center
        cr, cc = z['center']
        coord_map[(floor, cr, cc)] = (sid, 'center')
        # Slots
        for (sr, sc) in z['slots']:
            # 如果 slot 和 center 重疊 (有些設計會這樣)，優先算 center
            if (floor, sr, sc) not in coord_map:
                coord_map[(floor, sr, sc)] = (sid, 'slot')

    # 2. 填充時間軸
    print("   正在計算佔用矩陣...")
    for agv_id, group in agv_groups:
        if not str(agv_id).startswith('AGV'): continue
        group = group.sort_values('s_sec')
        records = group.to_dict('records')
        
        for i in range(len(records)):
            curr_e = records[i]
            stay_start = curr_e['e_sec']
            if i + 1 < len(records):
                stay_end = records[i+1]['s_sec']
            else:
                stay_end = int(max_sim_time)
            
            if stay_end <= stay_start: continue
            
            key = (curr_e['floor'], int(curr_e['ey']), int(curr_e['ex']))
            
            if key in coord_map:
                sid, p_type = coord_map[key]
                s_idx = s_map[sid]
                
                # Numpy 切片更新 (非常快)
                # 邊界檢查
                start = max(0, stay_start)
                end = min(timeline_len, stay_end)
                
                if end > start:
                    total_counts[s_idx, start:end] += 1
                    if p_type == 'center':
                        working_counts[s_idx, start:end] += 1

    # 3. 檢查違規
    print("\n📊 驗證結果分析:")
    print(f"{'Station':<10} | {'Max Work':<10} | {'Max Total':<10} | {'Result':<10}")
    print("-" * 50)
    
    fail_count = 0
    
    for sid in station_ids:
        s_idx = s_map[sid]
        
        max_work = np.max(working_counts[s_idx])
        max_total = np.max(total_counts[s_idx])
        
        # 驗證條件
        cond1 = (max_total < 5) # 預期 < 5
        cond2 = (max_work <= 1) # 預期 = 1 (或 0)
        
        status = "PASS"
        if not cond1 or not cond2:
            status = "FAIL"
            fail_count += 1
            
        print(f"{sid:<10} | {max_work:<10} | {max_total:<10} | {status}")
        
        if not cond1:
            # 找出違規時間點
            bad_times = np.where(total_counts[s_idx] >= 5)[0]
            if len(bad_times) > 0:
                print(f"   ⚠️ [Violation] Total >= 5 at {len(bad_times)} seconds. (e.g., t={bad_times[0]}s)")

        if not cond2:
            bad_times = np.where(working_counts[s_idx] > 1)[0]
            if len(bad_times) > 0:
                print(f"   ⚠️ [Violation] Work > 1 at {len(bad_times)} seconds. (e.g., t={bad_times[0]}s)")
                
    print("-" * 50)
    if fail_count == 0:
        print("🎉 完美！所有物理限制驗證通過 (Strict Check Passed)。")
    else:
        print(f"❌ 警告：發現 {fail_count} 個工作站違反物理限制。")
        print("   建議檢查：Ghost 機制是否在工作站範圍內觸發，導致 AGV 重疊。")

if __name__ == "__main__":
    main()