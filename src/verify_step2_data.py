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
        df = pd.read_csv(FILE_PATH, dtype=str)
        df.columns = [c.upper().strip() for c in df.columns]
        
        # 搜尋 SD 相關的路線
        # 假設 ROUTECD 或 PARTCUSTID 有用到 SDTC
        print("\n🔍 搜尋關鍵字 'SD' ...")
        
        # 檢查 ROUTECD 為 SDTC 的
        mask_route = df['ROUTECD'].str.contains('SD', na=False, case=False)
        # 檢查 PARTCUSTID 為 SDTC 的
        mask_cust = df['PARTCUSTID'].str.contains('SD', na=False, case=False)
        
        target_df = df[mask_route | mask_cust]
        
        if target_df.empty:
            print("⚠️ 班次表中找不到任何 'SD' 相關的設定！")
            print("   -> 推論：如果找不到，程式會走 'Default 23:59'，但您看到的是 17:30...")
            print("   -> 可能原因：訂單上的 ROUTECD 是別的代號？")
        else:
            print(f"✅ 找到 {len(target_df)} 筆設定：")
            print(target_df[['ROUTECD', 'PARTCUSTID', 'ORDERENDTIME']].to_string())
            
    except Exception as e:
        print(f"❌ 讀取錯誤: {e}")

if __name__ == "__main__":
    main()