import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# ==========================================
# 1. 路徑與環境設定
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_MASTER_DIR = os.path.join(BASE_DIR, 'data', 'master')
DATA_TRX_DIR = os.path.join(BASE_DIR, 'data', 'transaction')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

os.makedirs(DATA_TRX_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def read_csv_robust(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 找不到檔案: {path}")
    try:
        return pd.read_csv(path, dtype=str, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding='cp950')

def parse_int_time(val):
    s = str(val).strip().split('.')[0]
    if not s or s.lower() == 'nan': return None
    s = s.zfill(4)
    try:
        return datetime.strptime(s, "%H%M").time()
    except ValueError:
        return None

def load_schedule_map():
    path = os.path.join(DATA_MASTER_DIR, 'route_schedule_master.csv')
    if not os.path.exists(path): return {}
    df = read_csv_robust(path)
    sched_map = {}
    for _, row in df.iterrows():
        if 'ROUTECD' not in row or 'PARTCUSTID' not in row: continue
        key = (str(row['ROUTECD']).strip(), str(row['PARTCUSTID']).strip())
        t = parse_int_time(row['ORDERENDTIME'])
        if t:
            if key not in sched_map: sched_map[key] = []
            sched_map[key].append(t)
    for k in sched_map: sched_map[k].sort()
    return sched_map

# ==========================================
# 新增：庫存白名單載入函式
# ==========================================
def load_valid_inventory_set():
    print("   🔒 [Filter] 正在建立庫存白名單...")
    path = os.path.join(DATA_MASTER_DIR, 'item_inventory.csv')
    if not os.path.exists(path):
        print("   ⚠️ 警告：找不到庫存檔，無法進行過濾！")
        return None

    try:
        df = pd.read_csv(path, dtype=str, encoding='utf-8')
    except:
        df = pd.read_csv(path, dtype=str, encoding='cp950')
    
    df.columns = [c.upper().strip() for c in df.columns]
    
    col_part = next((c for c in df.columns if 'PART' in c), None)
    col_cell = next((c for c in df.columns if 'CELL' in c or 'LOC' in c), None)
    
    valid_parts = set()
    
    if col_part and col_cell:
        for _, row in df.iterrows():
            cell = str(row[col_cell]).strip()
            part = str(row[col_part]).strip()
            # 【關鍵規則】只有儲位長度 >= 9 (代表實體料架) 才是有效庫存
            if len(cell) >= 9:
                valid_parts.add(part)
                
    print(f"   ✅ 白名單建立完成：共 {len(valid_parts)} 種有效料號")
    return valid_parts


def assign_wave(dt, sched_times):
    t = dt.time()
    for cutoff in sched_times:
        if t <= cutoff:
            return datetime.combine(dt.date(), cutoff), False
    # 跨日
    next_day_cutoff = sched_times[0]
    return datetime.combine(dt.date() + timedelta(days=1), next_day_cutoff), True

# [NEW] Follow 截止時間計算邏輯
def calculate_follow_deadline(row):
    route_cd = str(row['ROUTECD']).upper().strip()
    dt = row['datetime']
    
    # 基礎截止時間: 下班前 (17:30)
    end_of_day = datetime.combine(dt.date(), datetime.strptime("17:30", "%H:%M").time())
    
    # 邏輯: SDTC 10:00 以前取得的訂單，需要在 11:00 以前做完
    if route_cd == 'SDTC':
        cutoff_10am = datetime.combine(dt.date(), datetime.strptime("10:00", "%H:%M").time())
        target_11am = datetime.combine(dt.date(), datetime.strptime("11:00", "%H:%M").time())
        
        if dt < cutoff_10am:
            return target_11am
        else:
            return end_of_day
            
    # 其他 (SDHN, 或其他 SD 開頭的 Route) -> 下班前做完
    return end_of_day

def main():
    print("🚀 [Step 2] 啟動任務生成 (邏輯更新: HC11排除 / Follow定義 / SDTC時效)...")
    
    sched_map = load_schedule_map()
    df_orders = read_csv_robust(os.path.join(DATA_TRX_DIR, 'historical_orders_ex.csv'))
    df_recv = read_csv_robust(os.path.join(DATA_TRX_DIR, 'historical_receiving_ex.csv'))

    # 前處理
    for df in [df_orders, df_recv]:
            df.columns = [c.upper().strip() for c in df.columns]
            # [MODIFIED] 資料源已移除 FRCD，PARTNO 即為完整料號
            df['PART_ID'] = df['PARTNO'].fillna('').astype(str)

    # 時間與欄位標準化
    df_orders['datetime'] = pd.to_datetime(df_orders['DATE'] + ' ' + df_orders['TIME'], errors='coerce')
    df_orders = df_orders.dropna(subset=['datetime']).copy()
    
    if 'TRANSCD' not in df_orders.columns: df_orders['TRANSCD'] = '4'
    if 'PARTCUSTID' not in df_orders.columns: df_orders['PARTCUSTID'] = ''
    if 'ROUTECD' not in df_orders.columns: df_orders['ROUTECD'] = ''
    
    df_orders['PARTCUSTID'] = df_orders['PARTCUSTID'].astype(str).str.strip().str.upper()
    df_orders['ROUTECD'] = df_orders['ROUTECD'].astype(str).str.strip().str.upper()

    # ==========================================
    # 邏輯 0: 幽靈訂單過濾 (Ghost Order Filter)
    # ==========================================
    valid_parts = load_valid_inventory_set()
    if valid_parts is not None:
        original_count = len(df_orders)
        # 只保留 PARTNO 在白名單內的訂單
        df_orders = df_orders[df_orders['PART_ID'].isin(valid_parts)].copy()
        filtered_count = len(df_orders)
        
        diff = original_count - filtered_count
        if diff > 0:
            print(f"   👻 根據庫存檔，已剔除 {diff} 筆無實體儲位的幽靈訂單！")
        else:
            print("   ✨ 資料庫存檢查完美，無幽靈訂單。")

    # ==========================================
    # 邏輯 1: 排除 HC11
    # ==========================================
    original_len = len(df_orders)
    df_orders = df_orders[df_orders['ROUTECD'] != 'HC11'].copy()
    filtered_len = len(df_orders)
    if original_len > filtered_len:
        print(f"   ✂️ 已排除 ROUTECD='HC11' 共 {original_len - filtered_len} 筆")

    # ==========================================
    # 核心分流邏輯 (Priority & Classification)
    # ==========================================
    
    # A. 急單 (Urgent) - 優先權最高 (TRANSCD 3, 8)
    mask_urgent = df_orders['TRANSCD'].isin(['3', '8'])
    
    # B. 副倉補充 (Replenishment)
    # 判斷標準: LEFT(PARTCUSTID, 2) == 'SD'
    mask_rep = df_orders['PARTCUSTID'].str.startswith('SD') & (~mask_urgent)
    
    # C. Follow 任務
    # 判斷標準: LEFT(ROUTECD, 2) == 'SD'
    # 注意: 需排除已歸類為 Urgent 或 Rep 的 (避免重複，雖然依定義應不重疊)
    mask_follow = df_orders['ROUTECD'].str.startswith('SD') & (~mask_urgent) & (~mask_rep)
    
    # D. 一般波次 (Standard)
    # 剩下的就是一般波次
    mask_standard = (~mask_urgent) & (~mask_rep) & (~mask_follow)

    # ==========================================
    # 資料處理與存檔
    # ==========================================
    
    # 1. Urgent
    df_urgent = df_orders[mask_urgent].copy()
    
    # 2. Replenishment (副倉)
    df_replenishment = df_orders[mask_rep].copy()
    
    # 3. Follow
    df_follow = df_orders[mask_follow].copy()
    if not df_follow.empty:
        # 計算截止時間 (SDTC 邏輯)
        df_follow['DEADLINE'] = df_follow.apply(calculate_follow_deadline, axis=1)
        
        # 標記類型 (SDTC 或 SDHN/Other) 方便後續統計
        df_follow['FOLLOW_TYPE'] = df_follow['ROUTECD'].apply(lambda x: 'TC1' if x == 'SDTC' else 'OTHER')
        
        # 批次處理 (每 20 筆一組)
        df_follow = df_follow.sort_values(by=['DEADLINE', 'datetime']) # 急的排前面
        df_follow['BATCH_INDEX'] = df_follow.groupby('ROUTECD').cumcount() // 20
        df_follow['BATCH_ID'] = df_follow['ROUTECD'] + "_B" + df_follow['BATCH_INDEX'].astype(str)

    # 4. Standard
    df_standard = df_orders[mask_standard].copy()
    wave_results = []
    for _, row in df_standard.iterrows():
        key = (str(row['ROUTECD']).strip(), str(row['PARTCUSTID']).strip())
        if key in sched_map:
            deadline, _ = assign_wave(row['datetime'], sched_map[key])
            wave_id = f"W_{deadline.strftime('%Y%m%d_%H%M')}"
            wave_results.append({'WAVE_ID': wave_id, 'DEADLINE': deadline})
        else:
            # 找不到班次表的預設為當日最晚
            def_dl = datetime.combine(row['datetime'].date(), datetime.strptime("23:59", "%H:%M").time())
            wave_results.append({'WAVE_ID': 'W_DEFAULT', 'DEADLINE': def_dl})
    
    if not df_standard.empty:
        df_standard = pd.concat([df_standard, pd.DataFrame(wave_results, index=df_standard.index)], axis=1)

    # 5. Inbound
    df_recv['datetime'] = pd.to_datetime(df_recv['DATE'] + ' ' + df_recv['TIME'], errors='coerce')
    df_inbound = df_recv.dropna(subset=['datetime']).copy()

    # ==========================================
    # 輸出與報告
    # ==========================================
    print("💾 正在寫入 CSV...")
    df_standard.to_csv(os.path.join(DATA_TRX_DIR, 'tasks_standard.csv'), index=False, encoding='utf-8-sig')
    df_urgent.to_csv(os.path.join(DATA_TRX_DIR, 'tasks_urgent.csv'), index=False, encoding='utf-8-sig')
    df_replenishment.to_csv(os.path.join(DATA_TRX_DIR, 'tasks_replenishment.csv'), index=False, encoding='utf-8-sig')
    df_follow.to_csv(os.path.join(DATA_TRX_DIR, 'tasks_follow.csv'), index=False, encoding='utf-8-sig')
    df_inbound.to_csv(os.path.join(DATA_TRX_DIR, 'tasks_inbound.csv'), index=False, encoding='utf-8-sig')

    # 驗證報告
    print("\n🔍 [Step 2 結果驗證]")
    print(f"   📦 Standard (一般波次): {len(df_standard)} 筆")
    print(f"   🔄 Replenishment (副倉): {len(df_replenishment)} 筆 (PARTCUSTID='SD...')")
    print(f"   🚛 Follow (路線SD):      {len(df_follow)} 筆 (ROUTECD='SD...')")
    print(f"   ⚡ Urgent (急單):        {len(df_urgent)} 筆")
    
    # 檢查 SDTC 11:00 截止邏輯是否生效
    if not df_follow.empty:
        sdtc_early = df_follow[(df_follow['ROUTECD'] == 'SDTC') & 
                               (df_follow['datetime'].dt.hour < 10)]
        if not sdtc_early.empty:
            sample = sdtc_early.iloc[0]
            print(f"   ✅ [檢查] SDTC 早班單 (下單 {sample['datetime'].strftime('%H:%M')}) -> Deadline: {sample['DEADLINE']}")
        else:
            print("   ℹ️ (本次資料無 10:00 前的 SDTC 訂單)")

if __name__ == "__main__":
    main()