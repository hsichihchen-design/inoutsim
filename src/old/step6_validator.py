import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
EVENTS_FILE = os.path.join(LOG_DIR, 'simulation_events.csv')
DATA_FILE = os.path.join(BASE_DIR, 'processed_sim_data.pkl')
# ----------------------------------------

def main():
    print("🔍 [Step 6] 啟動數據驗證器 (Validator)...")
    
    if not os.path.exists(EVENTS_FILE):
        print(f"❌ 找不到 Log 檔: {EVENTS_FILE}")
        return

    # 1. 讀取基礎資料 (為了知道工作站座標)
    with open(DATA_FILE, 'rb') as f:
        sim_data = pickle.load(f)
    
    stations = sim_data['stations']
    # 建立座標反查表: (floor, x, y) -> station_id
    coord_to_station = {}
    for sid, info in stations.items():
        pos = info['pos'] # (row, col)
        # 注意：Log 中的座標通常是 (x, y) = (col, row)
        # 我們統一轉成字串 key 比較保險
        key = f"{info['floor']}_{pos[1]},{pos[0]}" # x,y
        coord_to_station[key] = sid

    print(f"   已載入 {len(stations)} 個工作站座標資訊。")

    # 2. 讀取 Events Log
    df = pd.read_csv(EVENTS_FILE)
    df['start_ts'] = pd.to_datetime(df['start_time'])
    df['end_ts'] = pd.to_datetime(df['end_time'])
    
    # 將時間轉為相對於模擬開始的秒數 (假設第一筆是最早時間)
    base_time = df['start_ts'].min()
    df['start_sec'] = (df['start_ts'] - base_time).dt.total_seconds().astype(int)
    df['end_sec'] = (df['end_ts'] - base_time).dt.total_seconds().astype(int)
    
    # 3. 分析每台 AGV 的停留區間
    # 我們需要找出：AGV 到達某個站的時間點，以及它下次移動的時間點
    
    agv_visits = []
    
    agv_groups = df.groupby('obj_id')
    
    for agv_id, group in agv_groups:
        if not agv_id.startswith('AGV'): continue
        
        group = group.sort_values('start_sec')
        events = group.to_dict('records')
        
        for i in range(len(events)):
            e = events[i]
            
            # 檢查這個事件的「終點」是不是工作站
            floor = e['floor']
            ex, ey = int(e['ex']), int(e['ey'])
            key = f"{floor}_{ex},{ey}"
            
            if key in coord_to_station:
                station_id = coord_to_station[key]
                arrival_time = e['end_sec']
                
                # 尋找離開時間 (下一個事件的開始時間)
                departure_time = arrival_time # 預設如果沒下個事件，就是瞬間
                if i + 1 < len(events):
                    next_e = events[i+1]
                    departure_time = next_e['start_sec']
                
                duration = departure_time - arrival_time
                
                # 過濾掉只是路過的 (停留 < 1秒)
                if duration > 1:
                    agv_visits.append({
                        'station': station_id,
                        'agv': agv_id,
                        'enter': arrival_time,
                        'leave': departure_time,
                        'duration': duration
                    })

    df_visits = pd.DataFrame(agv_visits)
    
    if df_visits.empty:
        print("⚠️ 沒有偵測到任何 AGV 進站停留紀錄。")
        return

    print(f"   已分析 {len(df_visits)} 次進站行為。")
    print("-" * 60)

    # 4. 回答問題 2: 每台 AGV 停留多久 (Dwell Time)
    print("📊 [驗證 2] AGV 工作站平均停留時間 (Dwell Time):")
    dwell_stats = df_visits.groupby('station')['duration'].describe()[['count', 'mean', 'max']]
    dwell_stats['mean'] = dwell_stats['mean'].round(1)
    print(dwell_stats)
    print("-" * 60)

    # 5. 回答問題 1: 同一時間有多少 AGV (Concurrency)
    # 這是最難的部分，我們用時間軸掃描法
    print("📊 [驗證 1] 工作站同時佔用分析 (Max Queue):")
    
    station_occupancy = {} # {sid: [t0, t1, t2... occupancy count]}
    max_time = int(df_visits['leave'].max())
    
    # 初始化
    for sid in stations.keys():
        station_occupancy[sid] = np.zeros(max_time + 10)

    # 填入佔用數據 (Timeline fill)
    for _, row in df_visits.iterrows():
        sid = row['station']
        s, e = int(row['enter']), int(row['leave'])
        if s < e:
            station_occupancy[sid][s:e] += 1
            
    # 統計結果
    results = []
    for sid, timeline in station_occupancy.items():
        max_occ = np.max(timeline)
        avg_occ = np.mean(timeline[timeline > 0]) if np.sum(timeline) > 0 else 0
        
        # 找出擁塞時段 (如果同時 > 2台)
        congested_seconds = np.sum(timeline >= 3)
        
        results.append({
            'Station': sid,
            'Max_AGV': int(max_occ),
            'Avg_AGV': round(avg_occ, 1),
            'Congested_Secs': congested_seconds
        })
        
    res_df = pd.DataFrame(results).sort_values('Max_AGV', ascending=False)
    print(res_df.to_string(index=False))
    
    print("-" * 60)
    print("💡 解讀說明:")
    print("1. Max_AGV: 該工作站「最高峰」時，同時有幾台車停在那裡 (包含正在工作和排隊)。")
    print("2. Congested_Secs: 有幾秒鐘該站累積了 3 台以上的車 (可能造成堵塞)。")
    print("3. Duration Mean: 平均每台車耗費多少秒 (包含排隊 + 實際作業)。")

if __name__ == "__main__":
    main()