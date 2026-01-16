import os
import time
import csv
from datetime import datetime

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
KPI_FILE = os.path.join(LOG_DIR, 'simulation_kpi.csv')
EVENTS_FILE = os.path.join(LOG_DIR, 'simulation_events.csv')
# ----------------------------------------

def get_file_info(filepath):
    if not os.path.exists(filepath):
        return "❌ 檔案不存在", 0, 0
    
    stats = os.stat(filepath)
    size_mb = stats.st_size / (1024 * 1024)
    mod_time = datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    
    # 暴力算行數 (最準)
    line_count = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, _ in enumerate(f):
                line_count = i + 1
    except Exception as e:
        return f"⚠️ 讀取錯誤: {e}", 0, 0
        
    return mod_time, size_mb, line_count

def main():
    print("🕵️‍♂️ [Step 8] 檔案鑑識報告")
    print(f"   檢查路徑: {LOG_DIR}\n")

    # 1. 檢查 KPI 檔案
    print(f"📄 Target: {os.path.basename(KPI_FILE)}")
    mtime, size, lines = get_file_info(KPI_FILE)
    print(f"   🕒 最後修改: {mtime}")
    print(f"   💾 檔案大小: {size:.2f} MB")
    print(f"   📝 實際行數: {lines} (含標題)")
    
    expected = 3000
    actual = lines - 1 # 扣掉標題
    
    if actual == expected:
        print(f"   ✅ 數據吻合: Log說跑了 {expected}, 檔案裡也有 {actual} 筆。")
        print("      👉 問題可能出在 Step 7 讀取時被 Filter 掉了？")
    elif actual < expected:
        print(f"   ❌ 數據遺失: Log說跑了 {expected}, 但檔案只有 {actual} 筆。")
        print(f"      📉 遺失了 {expected - actual} 筆資料。")
        print("      👉 可能性：")
        print("         1. 檔案被其他程式(如Excel)鎖定，導致寫入失敗。")
        print("         2. 程式雖然印出 Log，但寫入磁碟時發生權限錯誤或緩衝區異常。")
    else:
        print(f"   ❓ 數據異常: 檔案裡的 ({actual}) 比 Log 說的 ({expected}) 還多？")

    print("-" * 40)

    # 2. 檢查 Events 檔案
    print(f"📄 Target: {os.path.basename(EVENTS_FILE)}")
    mtime, size, lines = get_file_info(EVENTS_FILE)
    print(f"   🕒 最後修改: {mtime}")
    print(f"   📝 實際行數: {lines}")

    if lines < 10000:
        print("   ⚠️ 警告: Event Log 行數過少，模擬可能沒有完整記錄移動軌跡。")

if __name__ == "__main__":
    main()