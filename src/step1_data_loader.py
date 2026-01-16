import pandas as pd
import numpy as np
import os
import sys

# 載入設定檔
try:
    import step0_config as config
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import step0_config as config

# 路徑設定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_MASTER_DIR = os.path.join(BASE_DIR, 'data', 'master')
DATA_MAPPING_DIR = os.path.join(BASE_DIR, 'data', 'mapping')
os.makedirs(DATA_MAPPING_DIR, exist_ok=True)

# 檔案名稱
MAP_2F_FILE = '2F_map.xlsx'
MAP_3F_FILE = '3F_map.xlsx'
ALL_CELL_LIST_FILE = 'all_cell_list.csv'
ITEM_INVENTORY_FILE = 'item_inventory.csv'

def load_excel_map(filename):
    path = os.path.join(DATA_MASTER_DIR, filename)
    if not os.path.exists(path):
        csv_path = path.replace('.xlsx', '.csv')
        if os.path.exists(csv_path): return pd.read_csv(csv_path, header=None).fillna(0).to_numpy()
        raise FileNotFoundError(f"找不到地圖檔: {path}")
    return pd.read_excel(path, header=None).fillna(0).to_numpy()

def get_shelf_coords(grid):
    rows, cols = grid.shape
    coords = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1: coords.append((r, c)) # 1=料架
    return sorted(coords, key=lambda x: (x[0], x[1]))

def main():
    print("🚀 [Step 1] 啟動資料載入...")

    # 1. 載入地圖
    grid_2f = load_excel_map(MAP_2F_FILE)
    grid_3f = load_excel_map(MAP_3F_FILE)
    coords_2f = get_shelf_coords(grid_2f)
    coords_3f = get_shelf_coords(grid_3f)

    print("🔍 驗證 Item Inventory (含 FRCD 識別)...")
    inv_path = os.path.join(DATA_MASTER_DIR, ITEM_INVENTORY_FILE)
    if os.path.exists(inv_path):
        df_inv = pd.read_csv(inv_path, dtype=str)
        df_inv.columns = [c.upper().strip() for c in df_inv.columns]
        
        # 尋找 FRCD 和 PARTNO
        col_frcd = next((c for c in df_inv.columns if 'FRCD' in c), None)
        col_part = next((c for c in df_inv.columns if 'PART' in c), None)
        
        if col_frcd and col_part:
            # 建立複合鍵
            df_inv['COMBO_ID'] = df_inv[col_frcd].fillna('') + df_inv[col_part].fillna('')
            print(f"✅ 已建立複合鍵 (FRCD+PARTNO)，範例: {df_inv['COMBO_ID'].iloc[0]}")
        else:
            print("⚠️ 警告: 庫存檔缺少 FRCD 或 PARTNO 欄位")
    


    # 2. 載入儲位清單 (修正欄位讀取邏輯)
    cell_path = os.path.join(DATA_MASTER_DIR, ALL_CELL_LIST_FILE)
    df_cells = pd.read_csv(cell_path, dtype=str)
    
    # [邏輯修正] 優先尋找 'ID' 欄位，其次找 'CELL'/'LOC'
    target_col = next((c for c in df_cells.columns if c.upper() == 'ID'), None)
    if not target_col:
        target_col = next((c for c in df_cells.columns if 'CELL' in c.upper() or 'LOC' in c.upper()), None)
    
    if not target_col:
        raise ValueError(f"❌ 無法在 {ALL_CELL_LIST_FILE} 找到儲位ID欄位 (預期: ID)")

    print(f"   -> 使用欄位 '{target_col}' 作為儲位 ID")
    
    # 3. 歸戶邏輯 (Cell -> Shelf)
    shelves_map = {'2F': {}, '3F': {}}
    
    for cell_id in df_cells[target_col].dropna():
        cell_id = str(cell_id).strip()
        if len(cell_id) < 9: continue
        
        shelf_id = cell_id[:9] # [邏輯] 取前9碼
        floor = '2F' if cell_id.startswith('2') else '3F'
        
        if shelf_id not in shelves_map[floor]:
            shelves_map[floor][shelf_id] = []
        shelves_map[floor][shelf_id].append(cell_id)

    # 4. 驗證與映射
    mapping_data = []
    validation_log = []

    for floor, coords, shelf_dict in [('2F', coords_2f, shelves_map['2F']), ('3F', coords_3f, shelves_map['3F'])]:
        needed = len(shelf_dict)
        available = len(coords)
        
        # [Validation] 容量檢核
        if needed > available:
            validation_log.append(f"❌ {floor} 空間不足! 需 {needed} 架，僅有 {available} 格地圖點位。")
        else:
            validation_log.append(f"✅ {floor} 容量檢查通過 (使用率: {needed}/{available})")
        
        sorted_shelves = sorted(list(shelf_dict.keys()))
        for i, sid in enumerate(sorted_shelves):
            if i < len(coords):
                r, c = coords[i]
                for cid in shelf_dict[sid]:
                    mapping_data.append({'cell_id': cid, 'shelf_id': sid, 'floor': floor, 'x': c, 'y': r})

    # 5. 輸出
    df_out = pd.DataFrame(mapping_data)
    df_out.to_csv(os.path.join(DATA_MAPPING_DIR, 'shelf_coordinate_map.csv'), index=False)
    
    print("\n🔍 [Validation Report]")
    for log in validation_log: print(f"   {log}")
    print(f"   -> 總映射儲位數: {len(df_out)}")

if __name__ == "__main__":
    main()