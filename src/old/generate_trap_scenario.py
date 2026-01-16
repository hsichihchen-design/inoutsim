import pandas as pd
import numpy as np
import pickle
import os
from collections import deque
from datetime import datetime

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_MAP_DIR = os.path.join(BASE_DIR, 'data', 'master')
MAPPING_DIR = os.path.join(BASE_DIR, 'data', 'mapping')
INPUT_FILE = os.path.join(BASE_DIR, 'processed_sim_data.pkl')

os.makedirs(DATA_MAP_DIR, exist_ok=True)
os.makedirs(MAPPING_DIR, exist_ok=True)

def generate_trap_scenario():
    print("🛠️ 正在生成「四面楚歌」測試場景...")

    # 1. 建立 32x61 的地圖 (0:空地, -1:牆壁, 1:儲位, 4:排隊區)
    # 我們做一個簡單的空曠房間，方便觀察
    grid_2f = np.zeros((32, 61), dtype=int)
    grid_3f = np.zeros((32, 61), dtype=int) # 3F 留空不用

    # 畫牆壁邊框
    grid_2f[0, :] = -1; grid_2f[-1, :] = -1
    grid_2f[:, 0] = -1; grid_2f[:, -1] = -1
    
    # 填充內部為可行走儲位區域 (1)
    grid_2f[1:31, 1:60] = 1

    # 設定工作站位置與排隊區
    STATION_POS = (15, 5) # 工作站在左側
    grid_2f[STATION_POS] = 0 # 工作站本身是空地
    # 工作站右邊設為排隊區
    grid_2f[15, 6] = 4
    grid_2f[15, 7] = 4

    # 2. 定義陷阱區 (The Trap)
    # Target 在 (15, 30)，被四個 Block 包圍
    TARGET_POS = (15, 30)
    BLOCK_UP    = (14, 30)
    BLOCK_DOWN  = (16, 30)
    BLOCK_LEFT  = (15, 29)
    BLOCK_RIGHT = (15, 31)

    shelf_coords = {}
    shelf_list = []

    # 建立 Target
    shelf_coords['SHELF_TARGET'] = {'floor': '2F', 'pos': TARGET_POS}
    shelf_list.append({'id': 'SHELF_TARGET', 'floor': '2F', 'x': TARGET_POS[1], 'y': TARGET_POS[0]})

    # 建立 Blockers
    blockers = {
        'BLOCK_UP': BLOCK_UP,
        'BLOCK_DOWN': BLOCK_DOWN,
        'BLOCK_LEFT': BLOCK_LEFT,   # 這是最可能被搬走的，因為它擋在去工作站的直線上
        'BLOCK_RIGHT': BLOCK_RIGHT
    }

    for name, pos in blockers.items():
        shelf_coords[name] = {'floor': '2F', 'pos': pos}
        shelf_list.append({'id': name, 'floor': '2F', 'x': pos[1], 'y': pos[0]})

    # 3. 為了讓視覺化好看，我們生成 Excel 地圖檔
    print(f"📄 輸出地圖檔至 {DATA_MAP_DIR} ...")
    df_map = pd.DataFrame(grid_2f)
    df_map.to_excel(os.path.join(DATA_MAP_DIR, '2F_map.xlsx'), header=False, index=False)
    
    # 3F 雖然不用但也生成一下避免報錯
    pd.DataFrame(grid_3f).to_excel(os.path.join(DATA_MAP_DIR, '3F_map.xlsx'), header=False, index=False)

    # 4. 生成 shelf_coordinate_map.csv (Visualizer 用)
    print(f"📄 輸出料架座標檔至 {MAPPING_DIR} ...")
    df_shelf = pd.DataFrame(shelf_list)
    df_shelf.to_csv(os.path.join(MAPPING_DIR, 'shelf_coordinate_map.csv'), index=False)

    # 5. 生成模擬數據 pickle
    stations = {
        'WS_TEST': {'floor': '2F', 'pos': STATION_POS}
    }

    # 建立一個唯一的任務：去搬 Target
    queues = {
        '2F': deque([
            {
                'shelf_id': 'SHELF_TARGET',
                'wave_id': 'TEST_WAVE',
                'stops': [{'station': 'WS_TEST', 'time': 10}]
            }
        ]),
        '3F': deque()
    }

    data = {
        'grid_2f': grid_2f,
        'grid_3f': grid_3f,
        'stations': stations,
        'shelf_coords': shelf_coords,
        'queues': queues,
        'base_time': datetime(2025, 1, 1, 8, 0, 0)
    }

    with open(INPUT_FILE, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✅ 測試數據已生成: {INPUT_FILE}")
    print("👉 請依序執行: step4_simulation_core.py -> step5_visualizer.py")

if __name__ == "__main__":
    generate_trap_scenario()