import pandas as pd
import numpy as np
import os
import random
from collections import defaultdict

# ==========================================
# 1. 路徑與環境設定
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_MASTER_DIR = os.path.join(BASE_DIR, 'data', 'master')
DATA_MAPPING_DIR = os.path.join(BASE_DIR, 'data', 'mapping')

class StrategyEngine:
    def __init__(self):
        print("🧠 [Strategy] 初始化高階策略引擎 (MinShelf 版)...")
        
        self.map_df = self._load_map()
        self.bin_specs = self._load_cell_category() 
        self.part_info = self._load_part_info()
        
        # inventory[pid][shelf_id] = {bin_id: qty}
        self.inventory = defaultdict(lambda: defaultdict(dict))
        self.part_floor_map = {} 
        self.shelf_stats = defaultdict(lambda: {'total_madfrq': 0, 'has_heavy': False, 'skus': set()})
        
        self._load_initial_inventory()
        print(f"   -> 策略引擎就緒: 已登錄 {len(self.part_floor_map)} 種零件")

    def _get_combo_id(self, frcd, partno):
        return str(frcd).strip() + str(partno).strip()
    
    def is_part_exist(self, frcd, partno):
        pid = self._get_combo_id(frcd, partno)
        return pid in self.part_floor_map

    def _load_cell_category(self):
        path = os.path.join(DATA_MASTER_DIR, 'cell_category.csv')
        specs = {}
        if os.path.exists(path):
            df = pd.read_csv(path, dtype=str)
            df.columns = [c.upper().strip() for c in df.columns]
            for _, r in df.iterrows():
                cat = r.get('CELLCAT')
                if cat:
                    specs[cat] = {
                        'D': float(r.get('D', 0)),
                        'W': float(r.get('W', 0)),
                        'H': float(r.get('H', 0))
                    }
        return specs

    def _load_map(self):
        path = os.path.join(DATA_MAPPING_DIR, 'shelf_coordinate_map.csv')
        df = pd.read_csv(path, dtype=str)
        df['LEVEL'] = df['cell_id'].apply(lambda x: x[-3] if len(x) >= 3 else 'A')
        if 'floor' not in df.columns and 'FLOOR' in df.columns:
            df['floor'] = df['FLOOR']
        if 'cellcat' not in df.columns:
            df['cellcat'] = '1' 
        return df

    def _load_part_info(self):
        path = os.path.join(DATA_MASTER_DIR, 'all_item_info.csv')
        info = {}
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, dtype=str, encoding='cp950')
            except:
                df = pd.read_csv(path, dtype=str, encoding='utf-8')
            
            df.columns = [c.upper().strip() for c in df.columns]
            for _, row in df.iterrows():
                pid = str(row.get('PARTNO', '')).strip()
                h_val = float(row.get('H', 0)) if 'H' in row else 10
                
                info[pid] = {
                    'min_bin': int(row.get('MINBINCOUNT', 1)),
                    'weight': float(row.get('WT', 0)) if pd.notna(row.get('WT')) else 0,
                    'madfrq': float(row.get('MADFRQ', 0)) if pd.notna(row.get('MADFRQ')) else 0,
                    'D': float(row.get('D', 0)),
                    'W': float(row.get('W', 0)),
                    'H': h_val
                }
        return info

    def _load_initial_inventory(self):
        path = os.path.join(DATA_MASTER_DIR, 'item_inventory.csv')
        if not os.path.exists(path): return
        
        try:
            df = pd.read_csv(path, dtype=str, encoding='utf-8')
        except:
            df = pd.read_csv(path, dtype=str, encoding='cp950')

        df.columns = [c.upper().strip() for c in df.columns]
        
        col_frcd = next((c for c in df.columns if 'FRCD' in c), None)
        col_part = next((c for c in df.columns if 'PART' in c), None)
        col_cell = next((c for c in df.columns if 'CELL' in c or 'LOC' in c), None)
        col_qty = next((c for c in df.columns if 'QTY' in c or 'STOCK' in c), None)
        col_floor = next((c for c in df.columns if 'FLOOR' in c), None)
        
        if col_part and col_cell:
            for _, row in df.iterrows():
                frcd_val = row[col_frcd] if col_frcd else ''
                part_val = row[col_part]
                pid = self._get_combo_id(frcd_val, part_val)
                cell = str(row[col_cell]).strip()
                qty = int(row[col_qty]) if col_qty and pd.notna(row[col_qty]) else 100 
                
                floor = '2F'
                if col_floor and pd.notna(row[col_floor]):
                    f_val = str(row[col_floor]).strip()
                    if '2' in f_val: floor = '2F'
                    elif '3' in f_val: floor = '3F'
                else:
                    if cell.startswith('3'): floor = '3F'
                    else: floor = '2F'
                
                self.part_floor_map[pid] = floor

                if len(cell) >= 9:
                    sid = cell[:9]
                    self.inventory[pid][sid][cell] = qty
                    
                    p_data = self.part_info.get(pid, {})
                    self.shelf_stats[sid]['total_madfrq'] += p_data.get('madfrq', 0)
                    self.shelf_stats[sid]['skus'].add(pid)
                    if p_data.get('weight', 0) > 10: 
                        self.shelf_stats[sid]['has_heavy'] = True

    def check_fit_and_utilization(self, part_dims, bin_dims, qty):
        pD, pW, pH = part_dims['D'], part_dims['W'], part_dims['H']
        bD, bW, bH = bin_dims['D'], bin_dims['W'], bin_dims['H']
        
        total_vol = pD * pW * pH * qty
        bin_vol = bD * bW * bH
        
        if total_vol > bin_vol: return False, 0 
        if pH > bH: return False, 0 

        fit_normal = (pD <= bD and pW <= bW)
        fit_rotate = (pD <= bW and pW <= bD)
        
        if fit_normal or fit_rotate:
            utilization = total_vol / bin_vol
            return True, utilization
        return False, 0

    # ==========================================
    # 核心邏輯 1: 進貨 (Inbound) - 修正計數邏輯
    # ==========================================
    def find_best_bin_for_inbound(self, frcd, partno, qty):
        pid = self._get_combo_id(frcd, partno)
        
        if pid not in self.part_floor_map:
            return None, "UNKNOWN_PART"
            
        target_floor = self.part_floor_map[pid]
        p_info = self.part_info.get(pid, {'min_bin': 1, 'weight': 0, 'D':10, 'W':10, 'H':10})
        
        # [CRITICAL] 這裡計算的是「持有料架數 (Shelf Count)」，而非儲位數
        current_shelves = list(self.inventory[pid].keys())
        current_shelf_count = len(current_shelves)
        min_bin_satisfied = current_shelf_count >= p_info['min_bin']

        target_plan = None 

        # 決策樹
        if not min_bin_satisfied:
            # 情況 1: 分散不足 -> 強制找「新料架」
            target_plan = self._find_new_bin(pid, qty, p_info, target_floor, exclude_shelves=current_shelves)
        
        if not target_plan:
            # 情況 2: 分散已滿足 -> 優先找舊位填滿
            target_plan = self._find_existing_bin(pid, qty, require_full_fit=True)
        
        if not target_plan:
            # 情況 3: 舊位填不滿 -> 拆單填滿
            best_existing = self._find_existing_bin(pid, qty, require_full_fit=False)
            if best_existing: return best_existing 
            
        if not target_plan:
             # 情況 4: 舊位全滿 -> 找新位 (互斥鎖解除)
             exclude = [] 
             target_plan = self._find_new_bin(pid, qty, p_info, target_floor, exclude_shelves=exclude)

        return target_plan if target_plan else (None, "FULL_WAREHOUSE")

    def _find_new_bin(self, pid, qty, p_info, target_floor, exclude_shelves=[]):
        # 1. 排除已佔用的 Cells
        occupied_cells = set()
        for p in self.inventory:
            for s in self.inventory[p]:
                for c in self.inventory[p][s]:
                    occupied_cells.add(c)
        
        # 2. 篩選候選清單
        potential_df = self.map_df[
            (~self.map_df['cell_id'].isin(occupied_cells)) & 
            (self.map_df['floor'] == target_floor) &
            (~self.map_df['shelf_id'].isin(exclude_shelves)) 
        ]
        
        if len(potential_df) == 0: return None

        if len(potential_df) > 500:
            potential_cells = potential_df.sample(500).to_dict('records')
        else:
            potential_cells = potential_df.to_dict('records')
        
        valid_options = []
        for cell in potential_cells: 
            cat = str(cell.get('cellcat', '1'))
            bin_dim = self.bin_specs.get(cat, {'D':100, 'W':100, 'H':100})
            
            fits, util = self.check_fit_and_utilization(p_info, bin_dim, qty)
            if fits:
                valid_options.append({
                    'shelf': cell['shelf_id'],
                    'cell': cell['cell_id'],
                    'level': cell['LEVEL'],
                    'utilization': util
                })
        
        if not valid_options: return None
        
        max_util = max(x['utilization'] for x in valid_options)
        best_util_opts = [x for x in valid_options if x['utilization'] >= max_util - 0.05]
        
        shelves = list(set(x['shelf'] for x in best_util_opts))
        chosen_shelf = random.choice(shelves)
        shelf_opts = [x for x in best_util_opts if x['shelf'] == chosen_shelf]
        
        is_heavy = p_info['weight'] > 10.0
        if is_heavy:
            level_a = [x for x in shelf_opts if x['level'] == 'A']
            if level_a: return (chosen_shelf, level_a[0]['cell'])
            return (chosen_shelf, shelf_opts[0]['cell'])
        else:
            shelf_opts.sort(key=lambda x: x['level'])
            return (chosen_shelf, shelf_opts[0]['cell'])

    def _find_existing_bin(self, pid, qty, require_full_fit):
        candidates = []
        for sid, bins in self.inventory[pid].items():
            for bid, current_q in bins.items():
                if require_full_fit:
                    candidates.append((sid, bid))
                else:
                    candidates.append((sid, bid))
        if candidates: return candidates[0]
        return None

    def find_stock_for_outbound(self, wave_demand, station_demand):
        needed_pids = list(station_demand.keys())
        valid_pids = [p for p in needed_pids if p in self.inventory]
        
        candidate_shelves = set()
        for pid in valid_pids:
            for sid in self.inventory[pid]:
                candidate_shelves.add(sid)
        
        scores = []
        for sid in candidate_shelves:
            score = 0
            if self.shelf_stats[sid]['has_heavy']: score += 500
            
            hits = 0
            for pid in valid_pids:
                if sid in self.inventory[pid] and self.inventory[pid][sid]:
                     hits += 1
            if hits >= 2: score += 1000 * hits 
            else: score += 100 * hits
                
            for pid in valid_pids:
                if sid in self.inventory[pid]:
                    shelf_qty = sum(self.inventory[pid][sid].values())
                    wave_qty = wave_demand.get(pid, 0)
                    st_qty = station_demand.get(pid, 0)
                    if shelf_qty >= wave_qty: score += 2000 
                    elif shelf_qty >= st_qty: score += 500  
            
            score += self.shelf_stats[sid]['total_madfrq'] * 0.1
            scores.append((sid, score))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scores]

if __name__ == "__main__":
    eng = StrategyEngine()
    print("✅ Step 3 (StrategyEngine) 載入成功")