import pandas as pd
import os

# CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(os.path.dirname(BASE_DIR), 'logs') 
CSV_PATH = os.path.join(LOG_DIR, 'simulation_events.csv')

def analyze_purple_army():
    print(f"🕵️‍♂️ 正在分析 Event Log: {CSV_PATH}")
    
    if not os.path.exists(CSV_PATH):
        print("❌ 找不到 simulation_events.csv")
        return

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"❌ 讀取失敗: {e}")
        return

    events = []
    
    for _, row in df.iterrows():
        t = row['start_time'] 
        # NORMAL: 正常取貨/卸貨
        if row['type'] == 'SHELF_LOAD':
            events.append({'t': t, 'change': 1, 'agv': row['obj_id'], 'type': 'NORMAL'})
        elif row['type'] == 'SHELF_UNLOAD':
            events.append({'t': t, 'change': -1, 'agv': row['obj_id'], 'type': 'NORMAL'})
        # RESCUE: 移庫
        elif row['type'] == 'SHUFFLE_LOAD':
            events.append({'t': t, 'change': 1, 'agv': row['obj_id'], 'type': 'RESCUE'})
        elif row['type'] == 'SHUFFLE_UNLOAD':
            events.append({'t': t, 'change': -1, 'agv': row['obj_id'], 'type': 'RESCUE'})

    # 依時間排序
    events.sort(key=lambda x: str(x['t']))

    current_purple = 0
    current_teal = 0
    max_purple = 0
    max_teal = 0    # [FIX] 補上初始化
    max_total = 0
    
    print("\n📊 時間軸重播分析：")
    for e in events:
        if e['type'] == 'NORMAL':
            current_purple += e['change']
        else:
            current_teal += e['change']
            
        total = current_purple + current_teal
        
        if current_purple > max_purple: max_purple = current_purple
        if current_teal > max_teal: max_teal = current_teal # [FIX] 紀錄最大移庫數
        if total > max_total: max_total = total
            
    print("-" * 30)
    print(f"🟣 最大同時「紫色」車數 (正常任務): {max_purple}")
    print(f"🟢 最大同時「Teal色」車數 (移庫任務): {max_teal}") 
    print(f"🚙 最大同時「載貨」總車數 (Total): {max_total}")
    print("-" * 30)

if __name__ == "__main__":
    analyze_purple_army()