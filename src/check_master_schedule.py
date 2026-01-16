import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 修正路徑指向上一層的 data/master
DATA_MASTER_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data', 'master')
FILE_PATH = os.path.join(DATA_MASTER_DIR, 'route_schedule_master.csv')

def main():
    print(f"🔍 檢查班次表: {FILE_PATH}")
    if not os.path.exists(FILE_PATH):
        print("❌ 找不到檔案")
        return

    try:
        # [FIX] 嘗試多種編碼
        try:
            df = pd.read_csv(FILE_PATH, dtype=str, encoding='utf-8')
        except UnicodeDecodeError:
            print("⚠️ UTF-8 失敗，嘗試 CP950 (Big5)...")
            df = pd.read_csv(FILE_PATH, dtype=str, encoding='cp950')

        df.columns = [c.upper().strip() for c in df.columns]
        
        # 搜尋 SD 相關的路線
        print("\n🔍 搜尋關鍵字 'SD' ...")
        
        mask_route = df['ROUTECD'].str.contains('SD', na=False, case=False)
        mask_cust = df['PARTCUSTID'].str.contains('SD', na=False, case=False)
        
        target_df = df[mask_route | mask_cust]
        
        if target_df.empty:
            print("⚠️ 班次表中找不到任何 'SD' 相關的設定！")
        else:
            print(f"✅ 找到 {len(target_df)} 筆設定：")
            print(target_df[['ROUTECD', 'PARTCUSTID', 'ORDERENDTIME']].to_string())
            
    except Exception as e:
        print(f"❌ 讀取錯誤: {e}")

if __name__ == "__main__":
    main()