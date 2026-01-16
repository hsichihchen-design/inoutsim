import pandas as pd
import numpy as np
import os
import pickle
import random
from collections import defaultdict
from datetime import datetime

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_FILE = os.path.join(BASE_DIR, 'processed_sim_data.pkl')

class Preprocessor:
    def __init__(self):
        print("🚀 [Preprocessor] 初始化資料處理模組...")
        self.grid_2f = self._load_map('2F_map.xlsx', 32, 61)
        self.grid_3f = self._load_map('3F_map.xlsx', 32, 61)
        self.shelf_coords = self._load_shelf_coords()
        self.inventory_map = self._load_inventory()
        
        # 建立站點資訊
        self.stations = self._init_stations()
        
    def _load_map(self, filename, rows, cols):
        path = os.path.join(DATA_DIR, 'master', filename)
        if not os.path.exists(path): path = path.replace('.xlsx', '.csv')
        try:
            if filename.endswith('.xlsx'): df = pd.read_excel(path, header=None)
            else: df = pd.read_csv(path, header=None)
            raw = df.iloc[0:rows, 0:cols].fillna(0).values
            grid = np.full((rows, cols), -1.0) # -1 代表牆壁
            r_in, c_in = min(raw.shape[0], rows), min(raw.shape[1], cols)
            grid[0:r_in, 0:c_in] = raw[0:r_in, 0:c_in]
            return grid
        except Exception as e:
            print(f"⚠️ 無法讀取地圖 {filename}: {e}")
            return np.full((rows, cols), 0)

    def _load_shelf_coords(self):
        path = os.path.join(DATA_DIR, 'mapping', 'shelf_coordinate_map.csv')
        coords = {}
        if os.path.exists(path):
            df = pd.read_csv(path)
            # [FIX] 強制轉大寫，避免 Key Error
            df.columns = [c.upper().strip() for c in df.columns]
            
            # 嘗試找正確的欄位名 (相容不同命名習慣)
            col_shelf = next((c for c in df.columns if 'SHELF' in c), 'SHELF_ID')
            col_floor = next((c for c in df.columns if 'FLOOR' in c), 'FLOOR')
            col_x = next((c for c in df.columns if c == 'X'), 'X')
            col_y = next((c for c in df.columns if c == 'Y'), 'Y')

            for _, r in df.iterrows():
                try:
                    coords[str(r[col_shelf])] = {'floor': r[col_floor], 'pos': (int(r[col_y]), int(r[col_x]))}
                except KeyError:
                    pass # 略過欄位對不上的 row
        return coords

    def _load_inventory(self):
        path = os.path.join(DATA_DIR, 'master', 'item_inventory.csv')
        inv = defaultdict(list)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, dtype=str)
                # [FIX] 關鍵修正：直接修改 DataFrame 的 columns 為全大寫
                df.columns = [c.upper().strip() for c in df.columns]
                
                cols = df.columns
                part_col = next((c for c in cols if 'PART' in c), None)
                cell_col = next((c for c in cols if 'CELL' in c or 'LOC' in c), None)
                
                if part_col and cell_col:
                    print(f"   -> Inventory 欄位對應: Part='{part_col}', Loc='{cell_col}'")
                    for _, r in df.iterrows():
                        val_part = r[part_col]
                        val_cell = r[cell_col]
                        if pd.notna(val_part) and pd.notna(val_cell):
                            inv[str(val_part).strip()].append(str(val_cell).strip())
                else:
                    print(f"⚠️ Inventory 欄位對應失敗。現有欄位: {cols}")
            except Exception as e:
                print(f"⚠️ 讀取 Inventory 失敗: {e}")

        return inv

    def _init_stations(self):
        sts = {}
        for floor, grid in [('2F', self.grid_2f), ('3F', self.grid_3f)]:
            rows, cols = grid.shape
            cnt = 0
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == 2: # 2 代表工作站
                        cnt += 1
                        sts[f"{floor}_{cnt}"] = {'floor': floor, 'pos': (r, c)}
        return sts

    def _load_and_consolidate_orders(self):
        print("📦 正在讀取並合併訂單 (Order Batching)...")
        tasks_raw = []
        
        # 1. 讀取 Outbound
        path_out = os.path.join(DATA_DIR, 'transaction', 'wave_orders.csv')
        if os.path.exists(path_out):
            try:
                df = pd.read_csv(path_out)
                # [FIX] 強制轉大寫
                df.columns = [c.upper().strip() for c in df.columns]
                
                date_col = next((c for c in df.columns if 'DATETIME' == c), None)
                if not date_col:
                    date_col = next((c for c in df.columns if 'DATE' in c), None)
                
                if date_col:
                    df['datetime'] = pd.to_datetime(df[date_col])
                    df = df.dropna(subset=['datetime'])
                    if 'LOC' not in df.columns: df['LOC'] = ''
                    tasks_raw.extend(df.to_dict('records'))
                else:
                    print(f"⚠️ wave_orders.csv 找不到時間欄位 (DATETIME)")
            except Exception as e:
                print(f"⚠️ 讀取 wave_orders 錯誤: {e}")
        
        # 2. 讀取 Inbound (Receiving)
        path_in = os.path.join(DATA_DIR, 'transaction', 'historical_receiving_ex.csv')
        if os.path.exists(path_in):
            try:
                df_in = pd.read_csv(path_in)
                # [FIX] 強制轉大寫
                df_in.columns = [c.upper().strip() for c in df_in.columns]
                
                cols = df_in.columns
                date_col = next((c for c in cols if 'DATE' in c), None)
                part_col = next((c for c in cols if 'ITEM' in c or 'PART' in c), None)
                
                if date_col and part_col:
                    df_in['datetime'] = pd.to_datetime(df_in[date_col])
                    df_in['PARTNO'] = df_in[part_col]
                    df_in['WAVE_ID'] = 'RECEIVING_' + df_in['datetime'].dt.strftime('%Y%m%d')
                    df_in['PARTCUSTID'] = 'REC_VENDOR'
                    if 'LOC' not in df_in.columns: df_in['LOC'] = ''
                    tasks_raw.extend(df_in.to_dict('records'))
            except Exception as e:
                print(f"⚠️ 讀取 historical_receiving 錯誤: {e}")
        
        if not tasks_raw:
            print("⚠️ 無任何訂單資料！")
            return {'2F': [], '3F': []}, datetime.now()

        tasks_raw.sort(key=lambda x: x['datetime'])
        base_time = tasks_raw[0]['datetime']
        
        # --- 智慧併單 (Consolidation) ---
        print("   -> 進行庫存匹配與併單運算...")
        part_shelf_map = {}
        valid_shelves = list(self.shelf_coords.keys())
        
        # 先掃描一次有 LOC 的，建立 PART -> LOC 的對應 (黏滯性)
        for t in tasks_raw:
            part = str(t.get('PARTNO', '')).strip()
            loc = str(t.get('LOC', '')).strip()
            if len(loc) >= 5: # 假設至少要有長度
                part_shelf_map[part] = loc 
        
        # 填補沒有 LOC 的訂單
        for t in tasks_raw:
            loc = str(t.get('LOC', '')).strip()
            if len(loc) < 5:
                part = str(t.get('PARTNO', '')).strip()
                if part in part_shelf_map:
                    t['LOC'] = part_shelf_map[part]
                elif part in self.inventory_map and self.inventory_map[part]:
                    chosen = self.inventory_map[part][0]
                    t['LOC'] = chosen
                    part_shelf_map[part] = chosen
                elif valid_shelves:
                    # 隨機分配一個假位置，避免當機 (格式: SHELF-FACE-BIN)
                    rand_shelf = random.choice(valid_shelves)
                    t['LOC'] = f"{rand_shelf}-A-01"

        # 轉換為 AGV 任務格式
        df_tasks = pd.DataFrame(tasks_raw)
        final_queues = {'2F': [], '3F': []}
        
        if 'WAVE_ID' not in df_tasks.columns:
            df_tasks['WAVE_ID'] = 'DEFAULT_WAVE'
            
        grouped = df_tasks.groupby('WAVE_ID')
        
        st_lists = {'2F': [k for k,v in self.stations.items() if v['floor']=='2F'],
                    '3F': [k for k,v in self.stations.items() if v['floor']=='3F']}
        
        for wave_id, wave_df in grouped:
            for floor in ['2F', '3F']:
                # 篩選屬於該樓層的訂單 (依據 LOC 開頭)
                # 假設 2F 的 loc 開頭是 '2'，3F 是 '3'
                prefix = floor[0]
                f_df = wave_df[wave_df['LOC'].str.startswith(prefix, na=False)].copy()
                
                if f_df.empty: continue
                
                avail_sts = st_lists[floor]
                if not avail_sts: continue
                
                # 依據貨架合併任務 (Batching by Shelf)
                shelf_tasks = defaultdict(list)
                for i, row in f_df.iterrows():
                    loc = str(row['LOC'])
                    # 假設 Shelf ID 是前 9 碼 (例如: 2F-01-01)
                    # 這裡做個防呆，如果長度不夠就整串當 ID
                    shelf_id = loc[:9] if len(loc) >= 9 else loc
                    
                    cust_id = str(row.get('PARTCUSTID', 'UNK'))
                    target_st = avail_sts[hash(cust_id) % len(avail_sts)]
                    
                    shelf_tasks[shelf_id].append({
                        'station': target_st,
                        'qty': row.get('QTY', 1),
                        'row': row
                    })
                
                # 生成最終任務物件
                for sid, items in shelf_tasks.items():
                    target_st = items[0]['station']
                    proc_time = 15 + (len(items) * 5)
                    
                    # 找出最早的時間當作任務時間
                    min_dt = min([x['row']['datetime'] for x in items])

                    task_obj = {
                        'task_id': f"{wave_id}_{sid}",
                        'type': 'ORDER',
                        'shelf_id': sid,
                        'wave_id': wave_id,
                        'priority': 10,
                        'stops': [{'station': target_st, 'time': proc_time}],
                        'datetime': min_dt,
                        'raw_items': [x['row'] for x in items]
                    }
                    final_queues[floor].append(task_obj)
                    
        # 排序
        for f in final_queues:
            final_queues[f].sort(key=lambda x: x['datetime'])
            
        return final_queues, base_time

    def run(self):
        queues, base_dt = self._load_and_consolidate_orders()
        
        data = {
            'grid_2f': self.grid_2f,
            'grid_3f': self.grid_3f,
            'stations': self.stations,
            'shelf_coords': self.shelf_coords,
            'queues': queues,
            'base_time': base_dt
        }
        
        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(data, f)
        print(f"✅ 資料處理完成！已儲存至 {OUTPUT_FILE}")
        print(f"   - 2F 任務數: {len(queues['2F'])}")
        print(f"   - 3F 任務數: {len(queues['3F'])}")

if __name__ == "__main__":
    Preprocessor().run()