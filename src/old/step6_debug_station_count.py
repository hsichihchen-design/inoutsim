import pickle
import os

# 設定路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, '..', 'processed_sim_data.pkl') # 假設在 src 目錄執行

def inspect_stations():
    print(f"🕵️‍♂️ 正在檢查資料檔: {INPUT_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        # 嘗試直接在當前目錄找
        INPUT_FILE_LOCAL = 'processed_sim_data.pkl'
        if os.path.exists(INPUT_FILE_LOCAL):
            path = INPUT_FILE_LOCAL
        else:
            print("❌ 找不到 .pkl 檔案，請確認路徑或先執行 step4_preprocessor.py")
            return
    else:
        path = INPUT_FILE

    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    stations = data['stations']
    st_2f = {k: v for k, v in stations.items() if v['floor'] == '2F'}
    
    print("\n" + "="*40)
    print(f"📊 2F 工作站統計結果")
    print(f"   預期數量: 8")
    print(f"   實際讀取數量: {len(st_2f)}")
    print("="*40)
    
    print("\n📍 詳細座標清單 (前 20 筆):")
    sorted_keys = sorted(st_2f.keys())
    for i, sid in enumerate(sorted_keys):
        pos = st_2f[sid]['pos']
        print(f"   {i+1}. ID: {sid} | 座標: {pos}")
        if i >= 19:
            print("   ... (還有更多)")
            break
            
    # 計算總容量
    print("-" * 40)
    print(f"⚠️ 系統判定的總容量 = {len(st_2f)} (站點) x 4 (佇列) = {len(st_2f) * 4} 台車")
    
    if len(st_2f) > 8:
        print("\n❌ [結論] 發生 '幽靈工作站' 現象！")
        print("   原因：Preprocessor 把工作站的每一個'格子'都當成了一個獨立的站點。")
        print("   後果：總容量暴增，ZoneManager 放行了過多的車輛。")
    else:
        print("\n✅ [結論] 站點數量正確 (8個)。問題可能出在 ZoneManager 邏輯本身。")

if __name__ == "__main__":
    inspect_stations()