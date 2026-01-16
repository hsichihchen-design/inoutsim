import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENTS_FILE = os.path.join(BASE_DIR, 'logs', 'simulation_events.csv')

def inspect_data():
    print("🔍 [數據法醫] 開始檢查 simulation_events.csv...")
    
    if not os.path.exists(EVENTS_FILE):
        print("❌ 找不到檔案！請先跑 Step 4。")
        return

    df = pd.read_csv(EVENTS_FILE)
    print(f"   -> 總筆數: {len(df)}")
    
    # 1. 檢查空值
    null_times = df['start_time'].isnull().sum() + df['end_time'].isnull().sum()
    if null_times > 0:
        print(f"⚠️ 警告：發現 {null_times} 筆時間為空 (NaN) 的資料！這會導致前端崩潰。")
    
    # 2. 檢查時間範圍 (轉換測試)
    # 使用 coerce 強制轉換，錯誤變成 NaT
    df['start_dt'] = pd.to_datetime(df['start_time'], errors='coerce')
    df['end_dt'] = pd.to_datetime(df['end_time'], errors='coerce')
    
    # 檢查 NaT (轉換失敗的日期)
    nat_count = df['start_dt'].isna().sum()
    if nat_count > 0:
        print(f"⚠️ 警告：有 {nat_count} 筆日期格式錯誤，無法轉換！")
        print("   -> 錯誤樣本:", df[df['start_dt'].isna()]['start_time'].head(3).values)

    # 剔除 NaT 後檢查範圍
    valid_df = df.dropna(subset=['start_dt', 'end_dt'])
    
    if not valid_df.empty:
        min_t = valid_df['start_dt'].min()
        max_t = valid_df['end_dt'].max()
        print(f"   📅 有效時間範圍: {min_t} ~ {max_t}")
        
        # 檢查是否有異常未來時間 (例如超過 7/2)
        outliers = valid_df[valid_df['end_dt'] > pd.Timestamp('2025-07-02')]
        if not outliers.empty:
            print(f"❌ 發現 {len(outliers)} 筆異常的未來數據 (超過 7/2)！")
            print("   -> 異常樣本:\n", outliers[['obj_id', 'start_time', 'end_time']].head(3))
        else:
            print("✅ 時間範圍在正常的一天內。")
    else:
        print("❌ 嚴重錯誤：沒有任何有效時間數據！")

if __name__ == "__main__":
    inspect_data()