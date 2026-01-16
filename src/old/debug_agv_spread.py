import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENTS_FILE = os.path.join(BASE_DIR, 'logs', 'simulation_events.csv')
COORD_FILE = os.path.join(BASE_DIR, 'data', 'mapping', 'shelf_coordinate_map.csv')

def analyze_spread():
    print("🔍 [AGV 分佈診斷] 開始分析...")

    # 1. 檢查目標座標定義 (Shelf Coordinates)
    if os.path.exists(COORD_FILE):
        df_coord = pd.read_csv(COORD_FILE)
        print(f"\n1. 座標映射表 (Target Definition):")
        print(f"   -> 總筆數: {len(df_coord)}")
        if not df_coord.empty:
            max_x = df_coord['x'].max()
            max_y = df_coord['y'].max()
            print(f"   -> 目標範圍: X(0~{max_x}), Y(0~{max_y})")
            if max_x < 10 and max_y < 10:
                print("   ⚠️ 警告：所有目標料架都擠在 (10,10) 以內！AGV 當然只會在那裡跑。")
            else:
                print("   ✅ 正常：目標料架分佈廣泛。")
    else:
        print("❌ 錯誤：找不到 shelf_coordinate_map.csv，AGV 沒有目標可去。")

    # 2. 檢查實際移動軌跡 (Simulation Events)
    if not os.path.exists(EVENTS_FILE):
        print("❌ 錯誤：找不到 simulation_events.csv，請先跑 Step 4。")
        return

    df = pd.read_csv(EVENTS_FILE)
    moves = df[df['type'] == 'AGV_MOVE']
    
    if moves.empty:
        print("❌ 錯誤：沒有任何移動事件 (AGV_MOVE)。")
        return

    print(f"\n2. 實際移動軌跡 (Simulation Results):")
    # 收集所有出現過的座標
    all_x = pd.concat([moves['sx'], moves['ex']])
    all_y = pd.concat([moves['sy'], moves['ey']])
    
    real_min_x, real_max_x = all_x.min(), all_x.max()
    real_min_y, real_max_y = all_y.min(), all_y.max()
    
    print(f"   -> X 軸範圍: {real_min_x} ~ {real_max_x}")
    print(f"   -> Y 軸範圍: {real_min_y} ~ {real_max_y}")
    
    width = real_max_x - real_min_x
    height = real_max_y - real_min_y
    
    print(f"   -> 活動區域大小: {width} x {height}")

    # 判定結論
    print("\n====== 診斷結論 ======")
    if real_max_x < 15 and real_max_y < 15:
        print("❌ 【模擬邏輯問題】AGV 被困在左上角！")
        print("   可能原因：")
        print("   1. 地圖讀取失敗，使用了 10x10 預設地圖。")
        print("   2. shelf_coordinate_map.csv 裡的座標全都是錯的。")
    else:
        print("✅ 【模擬數據正常】AGV 確實有跑遍全圖 (數值大於 15)。")
        print("   👉 如果您在畫面上看到它們擠在左上角，那是 Step 5 (Visualizer) 的 Canvas 縮放比例寫錯了。")

if __name__ == "__main__":
    analyze_spread()