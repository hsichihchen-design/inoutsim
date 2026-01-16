import pandas as pd
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_MAP_DIR = os.path.join(BASE_DIR, 'data', 'master')

def print_map_ascii(filename):
    print(f"\n🔍 正在解析地圖視覺結構: {filename}")
    path = os.path.join(DATA_MAP_DIR, filename)
    if not os.path.exists(path):
        print("❌ 檔案不存在")
        return

    # 讀取
    df = pd.read_excel(path, header=None).fillna(0)
    grid = df.values
    rows, cols = grid.shape
    
    print(f"   -> 尺寸: {rows} (列/高) x {cols} (欄/寬)")
    
    # 縮小顯示 (如果地圖太大，終端機會亂掉，我們每 N 格取樣一次，或者只印左上角)
    # 這裡我們嘗試印出完整結構，用符號代表
    
    print("\n--- [Python 眼中的地圖] (X=欄, Y=列) ---")
    print("   " + "".join([str(c%10) for c in range(min(cols, 60))])) # 尺標
    
    for r in range(min(rows, 40)): # 只印前 40 列以免洗版
        row_str = f"{r:02d} "
        for c in range(min(cols, 60)): # 只印前 60 欄
            val = grid[r][c]
            if val == 1:
                char = '▓' # 料架 (Shelf)
            elif val == 2:
                char = 'W' # 工作站 (Workstation)
            elif val == 3:
                char = 'C' # 充電站 (Charger)
            elif val == 0:
                char = '.' # 空地
            else:
                char = '?'
            row_str += char
        print(row_str)
        
    print("------------------------------------------")
    
    # 檢查工作站座標
    ws_coords = np.argwhere(grid == 2)
    print(f"\n📍 工作站座標 (Row, Col):")
    if len(ws_coords) > 0:
        for rc in ws_coords:
            print(f"   - Row(Y): {rc[0]}, Col(X): {rc[1]}")
    else:
        print("   ❌ 沒看到工作站 (W)！")

if __name__ == "__main__":
    print_map_ascii('2F_map.xlsx')