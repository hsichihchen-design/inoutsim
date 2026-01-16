import pandas as pd
import numpy as np
import os
import pickle
from collections import defaultdict

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
EVENTS_FILE = os.path.join(LOG_DIR, 'simulation_events.csv')
DATA_FILE = os.path.join(BASE_DIR, 'processed_sim_data.pkl')
# ----------------------------------------

def main():
    print("🕵️ [Step 10] 啟動任務重疊偵探 (Task Overlap Detective)...")
    
    if not os.path.exists(EVENTS_FILE):
        print("❌ 找不到 Log 檔。")
        return

    # 1. 載入工作站座標反查表
    with open(DATA_FILE, 'rb') as f:
        sim_data = pickle.load(f)
    
    # 建立 (floor, x, y) -> station_id
    coord_to_station = {}
    for sid, info in sim_data['stations'].items():
        pos = info['pos']
        key = (info['floor'], pos[0], pos[1]) # (row, col)
        coord_to_station[key] = sid

    # 2. 讀取事件
    df = pd.read_csv(EVENTS_FILE)
    df['start_ts'] = pd.to_datetime(df['start_time'])
    df['end_ts'] = pd.to_datetime(df['end_time'])
    base_time = df['start_ts'].min()
    df['s_sec'] = (df['start_ts'] - base_time).dt.total_seconds().astype(int)
    df['e_sec'] = (df['end_ts'] - base_time).dt.total_seconds().astype(int)

    # 3. 提取 "工作區間" (Work Intervals)
    # 邏輯：AGV 到達工作站 (End of Move) ~ 下一次移動開始 (Start of Next Move)
    station_intervals = defaultdict(list)
    
    for agv_id, group in df.groupby('obj_id'):
        if not str(agv_id).startswith('AGV'): continue
        
        records = group.sort_values('s_sec').to_dict('records')
        
        for i in range(len(records)):
            curr_e = records[i]
            
            # 檢查這個事件的終點是不是工作站
            # 注意 log 的 ex, ey 是 col, row
            key = (curr_e['floor'], int(curr_e['ey']), int(curr_e['ex']))
            
            if key in coord_to_station:
                sid = coord_to_station[key]
                
                # 進入時間 (抵達瞬間)
                enter_time = curr_e['e_sec']
                
                # 離開時間 (下一次移動開始)
                if i + 1 < len(records):
                    leave_time = records[i+1]['s_sec']
                else:
                    leave_time = enter_time + 20 # 假設最後停了20秒
                
                duration = leave_time - enter_time
                
                # 只有停留超過 1 秒才算是在工作
                if duration > 1:
                    station_intervals[sid].append({
                        'agv': agv_id,
                        'start': enter_time,
                        'end': leave_time,
                        'duration': duration
                    })

    # 4. 檢查重疊 (Collision Check)
    print("\n🔍 重疊分析報告 (Overlap Report):")
    print(f"{'Station':<10} | {'Total Tasks':<12} | {'Overlaps':<10} | {'Max Concurrent':<15}")
    print("-" * 60)
    
    total_overlaps = 0
    
    for sid in sorted(station_intervals.keys()):
        intervals = sorted(station_intervals[sid], key=lambda x: x['start'])
        
        overlap_count = 0
        max_concurrent = 0
        
        # 掃描時間軸計算重疊
        if not intervals: continue
        
        # 簡單的掃描線演算法
        timeline = []
        for task in intervals:
            timeline.append((task['start'], 1)) # 進入 +1
            timeline.append((task['end'], -1))  # 離開 -1
            
        timeline.sort(key=lambda x: (x[0], x[1])) # 時間一樣時，先離開再進入 (避免誤判)
        
        curr_concurrency = 0
        local_max = 0
        has_overlap = False
        
        for t, change in timeline:
            curr_concurrency += change
            if curr_concurrency > local_max:
                local_max = curr_concurrency
            if curr_concurrency > 1:
                has_overlap = True
        
        # 計算有多少對任務重疊 (這比較複雜，我們只算發生重疊的次數)
        # 這裡簡化：只要 Max Concurrent > 1 就是 Fail
        status = "FAIL ❌" if local_max > 1 else "PASS ✅"
        if local_max > 1: total_overlaps += 1
            
        print(f"{sid:<10} | {len(intervals):<12} | {status:<10} | {local_max:<15}")
        
        # 列出具體的重疊案例 (只列前 3 個)
        if local_max > 1:
            print(f"   ⚠️ 具體案例 (Evidence):")
            count = 0
            for i in range(len(intervals)):
                for j in range(i+1, len(intervals)):
                    t1 = intervals[i]
                    t2 = intervals[j]
                    
                    # 判斷重疊: Start1 < End2 AND Start2 < End1
                    if t1['start'] < t2['end'] and t2['start'] < t1['end']:
                        print(f"      🔴 {t1['agv']} ({t1['start']}~{t1['end']}) 重疊 {t2['agv']} ({t2['start']}~{t2['end']})")
                        count += 1
                        if count >= 3: break
                if count >= 3: break
            print("      ...")

    print("-" * 60)
    if total_overlaps > 0:
        print(f"❌ 結論：共有 {total_overlaps} 個工作站發生任務重疊。")
        print("   這證實了 Ghost/Force Entry 機制破壞了「一個一個做」的規則。")
    else:
        print("🎉 結論：完美！所有任務都是依序執行的。")

if __name__ == "__main__":
    main()