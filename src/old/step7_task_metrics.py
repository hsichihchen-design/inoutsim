import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
KPI_FILE = os.path.join(LOG_DIR, 'simulation_kpi.csv')
EVENTS_FILE = os.path.join(LOG_DIR, 'simulation_events.csv')
DATA_FILE = os.path.join(BASE_DIR, 'processed_sim_data.pkl')
# ----------------------------------------

def main():
    print("📊 [Step 7] 啟動任務績效分析 (Task Metrics)...")
    
    if not os.path.exists(KPI_FILE) or not os.path.exists(EVENTS_FILE):
        print("❌ 缺少 Log 檔案，請先執行模擬 (Step 4)。")
        return

    # 1. 載入原始訂單總量 (分母)
    with open(DATA_FILE, 'rb') as f:
        sim_data = pickle.load(f)
    
    total_orders_2f = len(sim_data['queues']['2F'])
    total_orders_3f = len(sim_data['queues']['3F'])
    total_orders = total_orders_2f + total_orders_3f
    
    # 2. 分析 KPI (完工數量 - 分子)
    df_kpi = pd.read_csv(KPI_FILE)
    completed_count = len(df_kpi)
    
    print("\n✅ 1. 完工率評估 (Completion Rate)")
    print(f"   總訂單數: {total_orders}")
    print(f"   已完成數: {completed_count}")
    print(f"   完工進度: {completed_count / total_orders * 100:.1f}%")
    
    if completed_count < total_orders:
        print(f"   ⚠️ 尚有 {total_orders - completed_count} 個任務未完成或模擬時間不足。")
    else:
        print("   🎉 所有任務已全數完成！")

    # 3. 計算任務週期時間 (Cycle Time)
    # 定義：從 SHELF_LOAD (取貨) 到 SHELF_UNLOAD (放回) 的時間差
    print("\n⏱️ 2. 任務耗時分析 (Task Cycle Time)")
    
    df_events = pd.read_csv(EVENTS_FILE)
    df_events['ts'] = pd.to_datetime(df_events['end_time'])
    base_ts = df_events['ts'].min()
    df_events['sec'] = (df_events['ts'] - base_ts).dt.total_seconds()
    
    # 篩選與搬運有關的事件
    moves = df_events[df_events['type'].isin(['SHELF_LOAD', 'SHELF_UNLOAD'])].copy()
    moves = moves.sort_values(['obj_id', 'sec'])
    
    task_durations = []
    
    # 針對每一台 AGV 追蹤它的搬運歷程
    for agv_id, group in moves.groupby('obj_id'):
        current_load_time = None
        current_shelf = None
        
        for _, row in group.iterrows():
            etype = row['type']
            shelf_id = str(row['text']) # 確保是字串
            t = row['sec']
            
            if etype == 'SHELF_LOAD':
                current_load_time = t
                current_shelf = shelf_id
            
            elif etype == 'SHELF_UNLOAD':
                if current_load_time is not None and current_shelf == shelf_id:
                    duration = t - current_load_time
                    
                    # 過濾掉時間太短的 (可能是原地調整或 Bug)
                    if duration > 10: 
                        task_durations.append({
                            'agv': agv_id,
                            'shelf': shelf_id,
                            'duration': int(duration),
                            'start_sec': int(current_load_time),
                            'end_sec': int(t)
                        })
                # Reset
                current_load_time = None
                current_shelf = None

    if not task_durations:
        print("   ⚠️ 無法計算週期時間 (可能是沒有完成完整的 Load-Unload 閉環)。")
        return

    df_tasks = pd.DataFrame(task_durations)
    
    # 統計數據
    avg_time = df_tasks['duration'].mean()
    max_time = df_tasks['duration'].max()
    min_time = df_tasks['duration'].min()
    p90_time = df_tasks['duration'].quantile(0.9)
    
    print(f"   分析樣本: {len(df_tasks)} 筆完整任務")
    print(f"   平均耗時: {avg_time:.1f} 秒")
    print(f"   中位數  : {df_tasks['duration'].median():.1f} 秒")
    print(f"   最短/最長: {min_time} 秒 / {max_time} 秒")
    print(f"   P90 (90%的任務都在此時間內): {p90_time:.1f} 秒")

    # 4. 進階：找出「拖油瓶」任務 (耗時最久的前 5 名)
    print("\n🐢 3. 耗時最久的 5 個任務 (可能被 Ghost 或 塞車 拖累)")
    slowest = df_tasks.sort_values('duration', ascending=False).head(5)
    print(slowest[['agv', 'shelf', 'duration', 'start_sec']].to_string(index=False))

    # 5. (選用) 繪製直方圖
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(df_tasks['duration'], bins=30, color='skyblue', edgecolor='black')
        plt.title('Task Cycle Time Distribution')
        plt.xlabel('Seconds (Load to Unload)')
        plt.ylabel('Frequency')
        plt.axvline(avg_time, color='red', linestyle='dashed', linewidth=1, label=f'Avg: {avg_time:.1f}s')
        plt.legend()
        plt.grid(True, alpha=0.3)
        output_img = os.path.join(LOG_DIR, 'task_duration_dist.png')
        plt.savefig(output_img)
        print(f"\n📈 分布圖已儲存: {output_img}")
    except:
        print("\n⚠️ 無法繪圖 (可能缺少 matplotlib)")

if __name__ == "__main__":
    main()