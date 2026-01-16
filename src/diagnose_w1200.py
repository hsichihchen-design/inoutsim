import pandas as pd
import os
import sys
from collections import defaultdict

# ==========================================
# 1. 使用與 Step 5 完全相同的路徑邏輯
# ==========================================
CURRENT_FILE_PATH = os.path.abspath(__file__)
SRC_DIR = os.path.dirname(CURRENT_FILE_PATH)
BASE_DIR = os.path.dirname(SRC_DIR)

DATA_TRX_DIR = os.path.join(BASE_DIR, 'data', 'transaction')
DATA_MASTER_DIR = os.path.join(BASE_DIR, 'data', 'master')

def verify_deadlock_root_cause():
    print(f"🕵️‍♂️ [偵探程式] 啟動位置: {SRC_DIR}")
    print(f"📂 資料根目錄 (BASE_DIR): {BASE_DIR}")
    
    # --- 1. 檢查檔案存在性 ---
    inv_path = os.path.join(DATA_MASTER_DIR, 'item_inventory.csv')
    task_path = os.path.join(DATA_TRX_DIR, 'tasks_standard.csv')
    
    if not os.path.exists(inv_path):
        print(f"❌ 找不到庫存檔! 請確認路徑: {inv_path}")
        return
    if not os.path.exists(task_path):
        print(f"❌ 找不到任務檔! 請確認路徑: {task_path}")
        return

    print("✅ 檔案路徑檢查 OK，開始載入資料...")

    # --- 2. 載入庫存 (模擬 Step 3 的 Key 生成邏輯) ---
    try:
        df_inv = pd.read_csv(inv_path, dtype=str)
    except:
        df_inv = pd.read_csv(inv_path, dtype=str, encoding='cp950')
        
    df_inv.columns = [c.upper().strip() for c in df_inv.columns]
    
    # 抓取欄位
    col_frcd = next((c for c in df_inv.columns if 'FRCD' in c), None)
    col_part = next((c for c in df_inv.columns if 'PART' in c), None)
    col_cell = next((c for c in df_inv.columns if 'CELL' in c or 'LOC' in c), None)
    
    print(f"   -> 庫存欄位對應: FRCD=[{col_frcd}], PART=[{col_part}], CELL=[{col_cell}]")
    
    # 建立「嚴格庫存清單」 (只有 len >= 9 才是真的能被揀貨的)
    valid_inventory_keys = set() # 存 Combo ID (FRCD+PART)
    raw_part_only_keys = set()   # 存純 PARTNO (用來比對是否因前綴導致對不上)
    
    for _, row in df_inv.iterrows():
        p_val = str(row[col_part]).strip()
        f_val = str(row[col_frcd]).strip() if col_frcd else ''
        
        combo_id = f_val + p_val # Step 3 的 Key
        
        cell = str(row[col_cell]).strip()
        # [關鍵] 模擬 Step 3: 只有正規儲位才算數
        if len(cell) >= 9:
            valid_inventory_keys.add(combo_id)
            raw_part_only_keys.add(p_val)

    print(f"   -> 有效庫存 SKU 總數: {len(valid_inventory_keys)}")

    # --- 3. 載入卡住的波次 (W_1200) ---
    try:
        df_tasks = pd.read_csv(task_path, dtype=str)
    except:
        df_tasks = pd.read_csv(task_path, dtype=str, encoding='cp950')

    df_tasks.columns = [c.upper().strip() for c in df_tasks.columns]
    
    target_wave = 'W_20250701_1200'
    df_1200 = df_tasks[df_tasks['WAVE_ID'] == target_wave]
    
    print(f"\n🌊 分析波次 {target_wave} (共 {len(df_1200)} 筆任務)...")
    
    if len(df_1200) == 0:
        print("⚠️ 警告: 該波次沒有任何任務! 請確認 CSV 內容。")
        return

    # --- 4. 交叉比對 (找出幽靈訂單) ---
    ghost_count = 0
    reason_breakdown = defaultdict(int)
    examples = []

    for _, row in df_1200.iterrows():
        # Step 5 讀取的是 CSV 裡的 PARTNO
        task_pid = str(row.get('PARTNO', '')).strip()
        
        # 模擬 Step 5: 如果它在庫存 Key 裡找不到
        # 注意：這裡假設 Step 5 讀進來的 task_pid 應該要等於 inventory 的 Key
        if task_pid in valid_inventory_keys:
            continue # Pass
            
        ghost_count += 1
        
        # 診斷原因
        if task_pid in raw_part_only_keys:
            # 庫存裡有這個 Part，但 Key 對不上 (代表庫存 Key 有加 FRCD 前綴)
            reason = "前綴不一致 (Prefix Mismatch)"
            detail = f"任務Part: '{task_pid}' vs 庫存Part: '{task_pid}' (但庫存Key可能有FRCD)"
        else:
            # 庫存裡完全沒這個 Part (或者都在暫存區 len<9)
            reason = "無有效庫存 (No Valid Stock)"
            detail = f"Part: '{task_pid}'"
            
        reason_breakdown[reason] += 1
        if len(examples) < 3: examples.append(detail)

    print("-" * 60)
    if ghost_count == 0:
        print("✅ 恭喜? 數據完全匹配。")
        print("👉 這代表問題 100% 出在 Step 5 的程式邏輯死鎖 (Defaultdict 誤導)，而非資料本身。")
    else:
        print(f"❌ 抓到了! 發現 {ghost_count} 筆「幽靈任務」會導致死鎖。")
        print("📊 原因分析:")
        for r, c in reason_breakdown.items():
            print(f"   - {r}: {c} 筆")
        
        print("\n📝 失敗範例 (前3筆):")
        for ex in examples:
            print(f"   -> {ex}")
            
    print("-" * 60)

if __name__ == "__main__":
    verify_deadlock_root_cause()