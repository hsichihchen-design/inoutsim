import random
import numpy as np

# ==========================================
# 1. 系統參數配置 (System Configuration)
# ==========================================
CONFIG = {
    # --- 模擬基礎設定 ---
    "SIMULATION_START_TIME": "08:00",
    
    # --- 揀貨時間參數 (秒) ---
    "PICK_TIME_NORMAL": 7.0,   # 一般揀貨
    "PICK_TIME_REPACK": 15.4,  # 需拆包 (Repack=1)
    
    # --- 時間分佈參數 (常態分佈) ---
    # 任務切換 / 首件抵達
    "TIME_MODE_SWITCH_ARRIVAL": {
        "mu": 50, "sigma": 15, "min": 10, "max": 90
    },
    # 同類型連續作業 (流水線)
    "TIME_SHELF_SWITCH_SAME_MODE": {
        "mu": 15, "sigma": 5, "min": 5, "max": 30
    },
    # 上架時間 (進貨用)
    "TIME_PUTAWAY_PER_BIN": {
        "mu": 30, "sigma": 8, "min": 15, "max": 60
    }
}
# ==========================================
# 2. 共用工具函式 (Utilities)
# ==========================================
def sample_time(param):
    val = random.gauss(param['mu'], param['sigma'])
    return int(max(param['min'], min(val, param['max'])))

# ==========================================
# 3. 自我驗證模組 (Validation)
# ==========================================
def validate_config():
    print("🔍 [Validation] 正在驗證參數分佈邏輯...")
    test_count = 1000
    errors = 0
    
    for key, param in CONFIG.items():
        if isinstance(param, dict) and 'mu' in param:
            samples = [sample_time(param) for _ in range(test_count)]
            min_s, max_s = min(samples), max(samples)
            avg_s = sum(samples) / len(samples)
            
            # 檢查是否越界
            if min_s < param['min'] or max_s > param['max']:
                print(f"   ❌ {key}: 抽樣越界! ({min_s} ~ {max_s})")
                errors += 1
            else:
                print(f"   ✅ {key}: Pass (Avg: {avg_s:.1f}, Range: {min_s}-{max_s})")
    
    if errors == 0:
        print("🎉 Config 驗證通過：所有時間參數皆符合分佈規範。\n")
    else:
        print(f"⚠️ Config 驗證失敗：發現 {errors} 個異常。\n")

if __name__ == "__main__":
    validate_config()