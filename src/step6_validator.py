import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta

# ==========================================
# 1. 環境設定
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_TRX_DIR = os.path.join(BASE_DIR, 'data', 'transaction')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# ==========================================
# 2. 資料載入器
# ==========================================
def load_data():
    print("🚀 [Validator] 正在讀取模擬數據...")
    
    tasks = {}
    for name in ['tasks_standard.csv', 'tasks_urgent.csv', 'tasks_follow.csv', 'tasks_inbound.csv']:
        path = os.path.join(DATA_TRX_DIR, name)
        if os.path.exists(path):
            try: df = pd.read_csv(path, dtype=str, encoding='utf-8')
            except: df = pd.read_csv(path, dtype=str, encoding='cp950')
            tasks[name.replace('.csv', '')] = df
    
    event_path = os.path.join(LOG_DIR, 'validation_events.csv')
    if os.path.exists(event_path):
        events_df = pd.read_csv(event_path)
    else:
        events_df = pd.DataFrame()
        print("⚠️ 警告: 找不到 validation_events.csv")

    trace_path = os.path.join(LOG_DIR, 'simulation_trace.json')
    if os.path.exists(trace_path):
        with open(trace_path, 'r', encoding='utf-8') as f:
            trace_data = json.load(f)
    else:
        print("❌ 錯誤: 找不到 simulation_trace.json")
        sys.exit(1)
        
    return tasks, events_df, trace_data

# ==========================================
# 3. 驗證模組
# ==========================================
class Validator:
    def __init__(self, tasks, events, trace):
        self.tasks = tasks
        self.events = events
        self.trace = trace
        self.report = []

    def run(self):
        self.verify_daily_stats()
        self.verify_wave_timeliness()
        self.verify_logic_mechanisms()
        self.print_report()

    def verify_daily_stats(self):
        self.report.append("\n📊 [Metric 1] 每日任務統計 (Daily Statistics)")
        self.report.append(f"{'Type':<20} {'Total':<10} {'Shipped':<10} {'Rate':<10}")
        self.report.append("-" * 55)
        
        input_counts = {k: len(v) for k, v in self.tasks.items()}
        
        type_map = {
            'tasks_standard': 'STANDARD',
            'tasks_urgent': 'URGENT',
            'tasks_follow': 'FOLLOW',
            'tasks_inbound': 'INBOUND'
        }
        
        if not self.events.empty:
            for csv_name, type_key in type_map.items():
                total = input_counts.get(csv_name, 0)
                
                log_keywords = []
                if type_key == 'STANDARD': log_keywords = ['P1_WAVE']
                elif type_key == 'URGENT': log_keywords = ['P2_URGENT']
                elif type_key == 'FOLLOW': log_keywords = ['P3_FOLLOW']
                elif type_key == 'INBOUND': log_keywords = ['P4_INBOUND', 'STRATEGY', 'INBOUND']
                
                completed = 0
                for kw in log_keywords:
                    completed += self.events[
                        (self.events['Action'].isin(['ASSIGN', 'INBOUND'])) & 
                        (self.events['Details'].str.contains(kw, na=False))
                    ].shape[0]
                
                rate = (completed / total * 100) if total > 0 else 0.0
                self.report.append(f"{csv_name:<20} {total:<10} {completed:<10} {rate:.1f}%")

    def verify_wave_timeliness(self):
        """2. 波次準時率驗證 (修正欄位讀取邏輯)"""
        self.report.append("\n⏱️ [Metric 2] 波次準時率 (Wave Timeliness)")
        
        df_std = self.tasks.get('tasks_standard')
        if df_std is None or 'WAVE_ID' not in df_std.columns:
            self.report.append("⚠️ 無法驗證：缺少 Standard Task 資料")
            return

        # [FIX] 優先讀取 WAVE_DEADLINE，若無才讀 DEADLINE
        deadline_col = 'WAVE_DEADLINE' if 'WAVE_DEADLINE' in df_std.columns else 'DEADLINE'
        
        if deadline_col not in df_std.columns:
             self.report.append(f"⚠️ 無法驗證：找不到截止時間欄位 (預期 WAVE_DEADLINE 或 DEADLINE)")
             return

        # 1. 取得 Deadline
        wave_deadlines = df_std.dropna(subset=[deadline_col]).groupby('WAVE_ID')[deadline_col].max().to_dict()
        
        # 2. 取得實際完成時間
        wave_finish_times = {}
        trace_df = pd.DataFrame(self.trace)
        
        if 'time' in trace_df.columns:
            trace_df['dt'] = pd.to_datetime(trace_df['time'], errors='coerce')
        else:
            self.report.append("❌ Trace Log 格式錯誤")
            return

        all_waves = set(wave_deadlines.keys())
        
        for wid in all_waves:
            finish_time = None
            is_started = False
            
            for _, snap in trace_df.iterrows():
                waves_status = snap.get('waves', {})
                count = waves_status.get(wid, 0)
                
                if count > 0: is_started = True
                
                # 若曾經開始過，且現在變成 0 (或消失)，視為完成
                if is_started and count == 0:
                    finish_time = snap['dt']
                    break
            
            wave_finish_times[wid] = finish_time
        
        # 3. 統計
        on_time_cnt = 0
        delayed_cnt = 0
        unfinished_cnt = 0
        details = []
        
        for wid, deadline_str in wave_deadlines.items():
            try:
                deadline = pd.to_datetime(deadline_str)
                if pd.isna(deadline): continue 
            except: continue
                
            actual = wave_finish_times.get(wid)
            is_valid_actual = (actual is not None) and (not pd.isna(actual))
            
            status = ""
            diff_min = 0
            
            if not is_valid_actual:
                status = "Unfinished"
                unfinished_cnt += 1
            elif actual <= deadline:
                status = "On-Time"
                on_time_cnt += 1
            else:
                status = "DELAYED"
                delayed_cnt += 1
                diff_min = (actual - deadline).total_seconds() / 60
            
            if status == "DELAYED" or status == "Unfinished":
                dl_str = deadline.strftime('%H:%M')
                ac_str = actual.strftime('%H:%M') if is_valid_actual else "未完成"
                diff_str = f"{diff_min:.1f}" if is_valid_actual else "-"
                details.append(f"   - {wid}: {status} (表定: {dl_str}, 實際: {ac_str}, 延遲: {diff_str} min)")

        self.report.append(f"   ✅ 準時波次: {on_time_cnt}")
        self.report.append(f"   ❌ 延遲波次: {delayed_cnt}")
        self.report.append(f"   🛑 未完成波次: {unfinished_cnt} (模擬結束時尚未做完)")
        
        if details:
            self.report.append("   ⚠️ 延遲/未完成明細 (Top 10):")
            for d in details[:10]: self.report.append(d)
            if len(details) > 10: self.report.append(f"     ... (還有 {len(details)-10} 筆)")

    def verify_logic_mechanisms(self):
        self.report.append("\n⚙️ [Metric 3] 機制邏輯檢核 (Logic Check)")
        
        lock_violation = 0
        for snap in self.trace:
            shelves_in_use = []
            for st in snap['stations']:
                if st['state'] == 'WORKING' and st.get('shelf'):
                    shelves_in_use.append(st['shelf'])
            if len(shelves_in_use) != len(set(shelves_in_use)):
                lock_violation += 1
        
        if lock_violation == 0:
            self.report.append("   ✅ 資源鎖定 (Locking): 通過")
        else:
            self.report.append(f"   ❌ 資源鎖定: 失敗 ({lock_violation} 次衝突)")
            
        if not self.events.empty:
            stockout_cnt = self.events[self.events['Action'].str.contains('STOCKOUT', na=False)].shape[0]
            overflow_cnt = self.events[self.events['Action'].str.contains('OVERFLOW', na=False)].shape[0]
            
            if stockout_cnt == 0 and overflow_cnt == 0:
                self.report.append("   ✅ 異常防護: 無異常 (Perfect Run)")
            else:
                self.report.append(f"   ⚠️ 異常防護: 觸發缺料 {stockout_cnt} 次, 爆倉 {overflow_cnt} 次")
        else:
            self.report.append("   (無事件日誌)")

    def print_report(self):
        report_text = "\n".join(self.report)
        print(report_text)
        with open(os.path.join(LOG_DIR, 'validation_report_summary.txt'), 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n📄 完整報告已儲存至: logs/validation_report_summary.txt")

if __name__ == "__main__":
    t, e, tr = load_data()
    val = Validator(t, e, tr)
    val.run()