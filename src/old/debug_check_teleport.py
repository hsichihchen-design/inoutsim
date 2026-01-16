import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(BASE_DIR, 'logs', 'simulation_events.csv')

def check_teleport():
    if not os.path.exists(LOG_FILE):
        print("❌ 找不到 simulation_events.csv，請先執行 Step 4。")
        return

    print("🔍 正在分析 AGV 移動連續性...")
    df = pd.read_csv(LOG_FILE)
    df = df[df['type'] == 'AGV_MOVE'].sort_values(['obj_id', 'start_time'])
    
    agvs = df['obj_id'].unique()
    teleport_count = 0
    
    for agv in agvs:
        agv_data = df[df['obj_id'] == agv]
        prev_end_pos = None
        prev_end_time = None
        
        for _, row in agv_data.iterrows():
            curr_start_pos = (row['sx'], row['sy'])
            
            if prev_end_pos:
                # 檢查：上一段的終點，是否等於這一段的起點？
                dist = abs(curr_start_pos[0] - prev_end_pos[0]) + abs(curr_start_pos[1] - prev_end_pos[1])
                
                # 容許誤差 1 格 (避免浮點數誤差)，超過 1.5 代表瞬移
                if dist > 1.5:
                    print(f"⚠️ [瞬移偵測] {agv}:")
                    print(f"   上一次結束於 {prev_end_pos} (Time: {prev_end_time})")
                    print(f"   這一次開始於 {curr_start_pos} (Time: {row['start_time']})")
                    print(f"   -> 瞬間跳躍距離: {dist:.2f} 格")
                    teleport_count += 1
            
            prev_end_pos = (row['ex'], row['ey'])
            prev_end_time = row['end_time']

    if teleport_count == 0:
        print("✅ AGV 路徑連續，無瞬移現象。")
    else:
        print(f"❌ 總共發現 {teleport_count} 次瞬移事件！這就是亂動的原因。")

if __name__ == "__main__":
    check_teleport()