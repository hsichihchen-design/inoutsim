import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# ==========================================
# 設定檔案路徑
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_MASTER_DIR = os.path.join(BASE_DIR, 'data', 'master')
DATA_TRANSACTION_DIR = os.path.join(BASE_DIR, 'data', 'transaction')

# 輸入檔案
ROUTE_SCHEDULE_FILE = 'route_schedule_master.csv'
HISTORICAL_ORDERS_FILE = 'historical_orders_ex.csv'
OUTPUT_WAVE_FILE = 'wave_orders.csv'

def read_csv_robust(file_path, dtype=None):
    """
    強健的 CSV 讀取函式，自動嘗試 utf-8 與 cp950 (Big5) 編碼
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到檔案: {file_path}")

    encodings = ['utf-8', 'cp950', 'big5', 'gbk']
    
    for enc in encodings:
        try:
            # 加入 low_memory=False 與 dtype 指定，提升讀取穩定度
            df = pd.read_csv(file_path, encoding=enc, dtype=dtype, low_memory=False)
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            raise e
            
    raise ValueError(f"無法讀取檔案 {os.path.basename(file_path)}，請確認編碼")

def parse_time_str(time_str):
    """
    將 '855' (int/str) 或 '08:55:00' 轉為 time object
    """
    s = str(time_str).strip()
    if not s or s.lower() == 'nan': return None
    
    if ':' in s:
        # 格式: 08:55:00 或 08:55
        try:
            return datetime.strptime(s, "%H:%M:%S").time()
        except ValueError:
            try:
                return datetime.strptime(s, "%H:%M").time()
            except ValueError:
                return None
    else:
        # 格式: 855 -> 08:55
        if s.isdigit():
            s = s.zfill(4)
            try:
                return datetime.strptime(s, "%H%M").time()
            except ValueError:
                return None
        return None

def load_route_schedule():
    path = os.path.join(DATA_MASTER_DIR, ROUTE_SCHEDULE_FILE)
    print(f"📖 正在讀取班次表: {ROUTE_SCHEDULE_FILE} ...")
    
    # 強制將 ROUTECD 和 PARTCUSTID 讀為字串，避免型別混亂
    df = read_csv_robust(path, dtype={'ROUTECD': str, 'PARTCUSTID': str})
    
    schedule_map = {}
    count = 0
    
    # 清洗資料
    df['ROUTECD'] = df['ROUTECD'].str.strip()
    df['PARTCUSTID'] = df['PARTCUSTID'].str.strip()
    
    # 移除空值
    df.dropna(subset=['ROUTECD', 'PARTCUSTID', 'ORDERENDTIME'], inplace=True)
    
    for _, row in df.iterrows():
        key = (row['ROUTECD'], row['PARTCUSTID'])
        t = parse_time_str(row['ORDERENDTIME'])
        
        if t:
            if key not in schedule_map: schedule_map[key] = []
            schedule_map[key].append(t)
            count += 1
            
    for k in schedule_map:
        schedule_map[k].sort()
        
    print(f"   -> 已建立 {len(schedule_map)} 組客戶班次規則 (共 {count} 個班次時間點)")
    return schedule_map

def assign_wave(order_datetime, schedule_times):
    order_time = order_datetime.time()
    
    for cutoff_time in schedule_times:
        if order_time <= cutoff_time:
            wave_dt = datetime.combine(order_datetime.date(), cutoff_time)
            return wave_dt, False
            
    # 跨日
    next_day = order_datetime.date() + timedelta(days=1)
    first_cutoff = schedule_times[0]
    wave_dt = datetime.combine(next_day, first_cutoff)
    return wave_dt, True

def main():
    print("🚀 [Step 2] 啟動訂單波次產生器 (資料清洗版)...")
    
    # 1. 載入班次表
    try:
        schedule_map = load_route_schedule()
    except Exception as e:
        print(f"❌ 班次表讀取錯誤: {e}")
        sys.exit(1)
        
    # 2. 載入歷史訂單
    orders_path = os.path.join(DATA_TRANSACTION_DIR, HISTORICAL_ORDERS_FILE)
    print(f"📖 正在讀取歷史訂單: {HISTORICAL_ORDERS_FILE} ...")
    
    try:
        # 強制讀取為字串，後續再轉型，確保資料完整
        df_orders = read_csv_robust(orders_path, dtype=str)
    except Exception as e:
        print(f"❌ 訂單檔讀取錯誤: {e}")
        sys.exit(1)
    
    original_count = len(df_orders)
    
    # --- 資料清洗 ---
    print("🧹 執行資料清洗...")
    # 1. 移除 ROUTECD 或 PARTCUSTID 為空的行 (解決 ,,,,,,, 的問題)
    df_orders.dropna(subset=['ROUTECD', 'PARTCUSTID', 'DATE', 'TIME'], inplace=True)
    
    # 2. 去除空白字元
    df_orders['ROUTECD'] = df_orders['ROUTECD'].str.strip()
    df_orders['PARTCUSTID'] = df_orders['PARTCUSTID'].str.strip()
    
    # 3. 解析時間
    try:
        # 錯誤的時間格式轉為 NaT
        df_orders['datetime'] = pd.to_datetime(df_orders['DATE'] + ' ' + df_orders['TIME'], errors='coerce')
        # 移除時間解析失敗的行
        df_orders.dropna(subset=['datetime'], inplace=True)
    except Exception as e:
        print(f"❌ 時間格式解析嚴重錯誤: {e}")
        sys.exit(1)

    cleaned_count = len(df_orders)
    print(f"   -> 原始筆數: {original_count}, 清洗後有效筆數: {cleaned_count} (剔除 {original_count - cleaned_count} 筆無效資料)")

    if cleaned_count == 0:
        print("❌ 錯誤: 清洗後沒有剩餘任何訂單！請檢查 CSV 內容格式。")
        sys.exit(1)

    # 3. 進行波次分派
    print(f"⚙️ 開始分配波次...")
    
    wave_ids = []
    wave_timestamps = []
    is_next_day_list = []
    
    unmatched_keys = set()
    unmatched_count = 0
    
    for _, row in df_orders.iterrows():
        key = (row['ROUTECD'], row['PARTCUSTID'])
        
        if key in schedule_map:
            target_dt, is_next_day = assign_wave(row['datetime'], schedule_map[key])
            w_id = f"W_{target_dt.strftime('%Y%m%d_%H%M')}"
            
            wave_ids.append(w_id)
            wave_timestamps.append(target_dt)
            is_next_day_list.append(1 if is_next_day else 0)
        else:
            unmatched_count += 1
            if len(unmatched_keys) < 10: unmatched_keys.add(str(key))
            
            # Default Wave: 當日 23:59
            def_dt = datetime.combine(row['datetime'].date(), datetime.strptime("23:59", "%H:%M").time())
            wave_ids.append(f"W_{def_dt.strftime('%Y%m%d')}_DEFAULT")
            wave_timestamps.append(def_dt)
            is_next_day_list.append(0)
            
    df_orders['WAVE_ID'] = wave_ids
    df_orders['WAVE_DEADLINE'] = wave_timestamps
    df_orders['IS_ROLLOVER'] = is_next_day_list
    
    df_orders = df_orders.sort_values(by=['WAVE_DEADLINE', 'datetime'])
    
    output_path = os.path.join(DATA_TRANSACTION_DIR, OUTPUT_WAVE_FILE)
    df_orders.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ 波次生成完成！結果已存檔: {OUTPUT_WAVE_FILE}")
    print("\n📊 波次統計摘要:")
    print(f"   -> 有效訂單數: {cleaned_count}")
    print(f"   -> 生成波次數: {df_orders['WAVE_ID'].nunique()}")
    
    if unmatched_count > 0:
        print(f"   ⚠️ 警告: 有 {unmatched_count} 筆訂單找不到對應班次 (歸入 DEFAULT)")
        print(f"   🔍 找不到班次的 (Route, Cust) 範例: {list(unmatched_keys)}")
        print("      (請確認 route_schedule_master.csv 是否包含這些組合)")
        
    print("\n   [範例波次分佈 (前 5 筆)]:")
    print(df_orders[['WAVE_ID', 'ROUTECD', 'PARTCUSTID', 'datetime']].head(5).to_string())

if __name__ == "__main__":
    main()