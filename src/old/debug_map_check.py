import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def check_maps():
    print("🔍 檢查地圖檔案與讀取狀態...")
    
    # 檢查 2F
    path_2f = os.path.join(BASE_DIR, 'data', 'master', '2F_map.xlsx')
    if os.path.exists(path_2f):
        try:
            df = pd.read_excel(path_2f, header=None)
            grid = df.fillna(0).values
            print(f"✅ 2F 地圖讀取成功！大小: {grid.shape}")
            print(f"   -> 內容預覽 (Top Left 5x5):\n{grid[:5, :5]}")
        except Exception as e:
            print(f"❌ 2F 地圖存在但讀取失敗: {e}")
    else:
        print(f"❌ 找不到 2F 地圖檔案: {path_2f}")

    # 檢查 3F
    path_3f = os.path.join(BASE_DIR, 'data', 'master', '3F_map.xlsx')
    if os.path.exists(path_3f):
        try:
            df = pd.read_excel(path_3f, header=None)
            grid = df.fillna(0).values
            print(f"✅ 3F 地圖讀取成功！大小: {grid.shape}")
        except Exception as e:
            print(f"❌ 3F 地圖存在但讀取失敗: {e}")
    else:
        print(f"❌ 找不到 3F 地圖檔案: {path_3f}")

if __name__ == "__main__":
    check_maps()