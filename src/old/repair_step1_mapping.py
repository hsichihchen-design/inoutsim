import pandas as pd
import numpy as np
import os
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_MAP_DIR = os.path.join(BASE_DIR, 'data', 'master')
DATA_TRX_DIR = os.path.join(BASE_DIR, 'data', 'transaction')
OUTPUT_MAP_FILE = os.path.join(BASE_DIR, 'data', 'mapping', 'shelf_coordinate_map.csv')

def load_map_grid(filename):
    path = os.path.join(DATA_MAP_DIR, filename)
    if os.path.exists(path):
        try: return pd.read_excel(path, header=None).fillna(0).values
        except: pass
    # Try CSV
    csv_path = os.path.join(DATA_MAP_DIR, os.path.splitext(filename)[0] + ".csv")
    if os.path.exists(csv_path):
        try: return pd.read_csv(csv_path, header=None).fillna(0).values
        except: pass
    return None

def repair_mapping():
    print("🔧 啟動座標映射修復工具 (Mapping Repair)...")
    
    # 1. 讀取真實地圖上的料架點
    shelf_spots = {'2F': [], '3F': []}
    
    for floor, filename in [('2F', '2F_map.xlsx'), ('3F', '3F_map.xlsx')]:
        grid = load_map_grid(filename)
        if grid is None:
            print(f"   ❌ 無法讀取 {floor} 地圖")
            continue
            
        rows, cols = grid.shape
        count = 0
        # 掃描所有格子
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1: # 1 = 料架
                    # 注意：這裡存入 (x=Col, y=Row) 以符合視覺化習慣
                    shelf_spots[floor].append((c, r)) 
                    count += 1
        print(f"   ✅ {floor} 地圖中找到 {count} 個實體料架格")

    if not shelf_spots['2F'] and not shelf_spots['3F']:
        print("❌ 嚴重錯誤：地圖上完全沒有料架 (數值 1)！無法修復。")
        return

    # 2. 讀取所有訂單中出現過的料架 ID
    # 這裡我們需要一個清單，如果沒有清單，我們就讀 wave_orders.csv 來收集
    print("📦 收集訂單中的料架 ID...")
    order_path = os.path.join(DATA_TRX_DIR, 'wave_orders.csv')
    if not os.path.exists(order_path):
        print("❌ 找不到訂單檔 wave_orders.csv")
        return

    try:
        df_orders = pd.read_csv(order_path, encoding='utf-8-sig')
    except:
        df_orders = pd.read_csv(order_path, encoding='cp950') # big5 fallback

    # 假設訂單中有 shelf_id 欄位，如果沒有，我們就用 Row Index 當作假 ID
    # 但通常您的資料源應該隱含了料架資訊。
    # 為了保險，我們重新生成一份對應表。
    
    # 策略：我們產生足夠多的虛擬 ID，或者重置現有的 map
    # 讓我們讀取舊的 map 來獲取 ID 清單 (如果有的話)
    old_map_path = OUTPUT_MAP_FILE
    shelf_ids = []
    
    if os.path.exists(old_map_path):
        print("   -> 從舊的 mapping 檔讀取 ID...")
        try:
            df_old = pd.read_csv(old_map_path)
            shelf_ids = df_old['shelf_id'].unique().tolist()
        except:
            pass
    
    if len(shelf_ids) < 100:
        print("   -> 舊 ID 太少，從訂單生成虛擬 ID...")
        # 假設訂單有 PARTCUSTID，我們把它當作一種 ID，或者直接生成流水號
        shelf_ids = [f"Shelf_{i}" for i in range(len(df_orders))]
    
    print(f"   -> 準備重新分配 {len(shelf_ids)} 個料架 ID 位置...")

    # 3. 分配座標 (Round-Robin)
    new_rows = []
    
    # 混合 2F 和 3F 的空位
    all_spots = []
    for pos in shelf_spots['2F']: all_spots.append(('2F', pos))
    for pos in shelf_spots['3F']: all_spots.append(('3F', pos))
    
    if not all_spots:
        print("❌ 無處可放！")
        return
        
    random.shuffle(all_spots) # 洗牌，讓分佈更均勻
    
    for i, sid in enumerate(shelf_ids):
        # 輪詢分配
        floor, (x, y) = all_spots[i % len(all_spots)]
        new_rows.append({
            'shelf_id': sid,
            'floor': floor,
            'x': x,
            'y': y
        })
        
    # 4. 存檔
    df_new = pd.DataFrame(new_rows)
    df_new.to_csv(OUTPUT_MAP_FILE, index=False, encoding='utf-8')
    print(f"✅ 修復完成！已儲存至 {OUTPUT_MAP_FILE}")
    print("   -> 請重新執行 Step 4 (模擬) 與 Step 5 (視覺化)")

if __name__ == "__main__":
    repair_mapping()