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
LOG_DIR = os.path.join(BASE_DIR, 'logs')
DATA_TRX_DIR = os.path.join(BASE_DIR, 'data', 'transaction')

class SimulationValidator:
    def __init__(self):
        print("🔍 [Validator] 啟動模擬驗證程序...")
        self.trace_log = self._load_json_log('simulation_trace.json')
        self.event_log = self._load_csv_log('validation_events.csv')
        self.std_tasks = self._load_csv_data('tasks_standard.csv')
        
    def _load_json_log(self, fname):
        path = os.path.join(LOG_DIR, fname)
        if not os.path.exists(path):
            print(f"❌ 找不到 {fname}")
            return []
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_csv_log(self, fname):
        path = os.path.join(LOG_DIR, fname)
        if not os.path.exists(path): return pd.DataFrame()
        return pd.read_csv(path)

    def _load_csv_data(self, fname):
        path = os.path.join(DATA_TRX_DIR, fname)
        if not os.path.exists(path): return pd.DataFrame()
        try: df = pd.read_csv(path, dtype=str, encoding='utf-8')
        except: df = pd.read_csv(path, dtype=str, encoding='cp950')
        df.columns = [c.upper().strip() for c in df.columns]
        return df

    # ==========================================
    # 驗證 1: 產能與完工率 (Completion Rate)
    # ==========================================
    def check_throughput(self):
        print("\n📊 [1. 產能與完工率檢查]")
        if not self.trace_log: return

        # 讀取最後一筆快照的 KPI
        last_snap = self.trace_log[-1]
        kpi = last_snap.get('kpi', {})
        
        shipped = kpi.get('shipped', 0)
        received = kpi.get('received', 0)
        stockouts = kpi.get('stockouts', 0)
        
        # 估算總任務數 (從 CSV 檔案行數估算)
        total_std = len(self.std_tasks)
        # 這裡僅估算 Standard，若要精準需讀取所有 CSV
        
        print(f"   -> 最終出貨量 (Shipped): {shipped} 筆訂單")
        print(f"   -> 最終進貨量 (Received): {received} 筆任務")
        print(f"   -> 缺料次數 (Stockouts): {stockouts}")
        
        if total_std > 0:
            rate = (shipped / total_std) * 100
            print(f"   -> 一般波次完工率估算: {rate:.2f}% (基於 {total_std} 筆原始需求)")
            
            if rate < 99:
                print("   ⚠️ 警告: 完工率未達 100%，可能有任務卡在佇列中未消化完畢。")
            else:
                print("   ✅ 恭喜: 任務幾乎全數消化完畢。")

    # ==========================================
    # 驗證 2: 波次延遲分析 (Wave Delay)
    # ==========================================
    def analyze_wave_delays(self):
        print("\n⏱️ [2. 波次延遲分析]")
        if self.event_log.empty or self.std_tasks.empty:
            print("   ⚠️ 無法執行: 缺少 Event Log 或 原始任務檔")
            return

        # 1. 建立波次截止時間表 (Wave Deadline Map)
        # 需解析 tasks_standard.csv 中的 WAVE_DEADLINE
        wave_deadlines = {}
        for _, row in self.std_tasks.iterrows():
            wid = row.get('WAVE_ID')
            dl_str = row.get('DEADLINE') # 假設 Step 2 輸出欄位名是 DEADLINE 或 WAVE_DEADLINE
            if not dl_str: dl_str = row.get('WAVE_DEADLINE')
            
            if wid and dl_str:
                try:
                    # 嘗試解析多種格式
                    dl_dt = pd.to_datetime(dl_str)
                    # 只需要存一次 (假設同波次截止時間相同)
                    if wid not in wave_deadlines:
                        wave_deadlines[wid] = dl_dt
                except: pass
        
        print(f"   -> 已載入 {len(wave_deadlines)} 個波次的表定截止時間")

        # 2. 從 Event Log 找出每個波次的「最後派單時間」
        # 篩選 Category=DISPATCH, Action=ASSIGN
        # Details 格式範例: "Station 2F_ST_1 assigned P1_WAVE_W_20250701_0900 | Shelf: ..."
        dispatch_evts = self.event_log[
            (self.event_log['Category'] == 'DISPATCH') & 
            (self.event_log['Action'] == 'ASSIGN')
        ].copy()

        # 解析 WAVE_ID
        # 邏輯: 尋找字串中 P1_WAVE_ 開頭的部分
        def extract_wave_id(detail_str):
            if 'P1_WAVE_' in detail_str:
                # 假設格式: ... assigned P1_WAVE_{WID} | ...
                # 切割出 P1_WAVE_ 之後的字串，直到空格或 |
                try:
                    part = detail_str.split('P1_WAVE_')[1]
                    wid = part.split(' ')[0].split('|')[0]
                    return wid
                except: return None
            return None

        dispatch_evts['WAVE_ID'] = dispatch_evts['Details'].apply(extract_wave_id)
        dispatch_evts = dispatch_evts.dropna(subset=['WAVE_ID'])
        
        # 加上日期 (模擬日)
        sim_date = "2025-07-01" # 需與模擬一致
        dispatch_evts['datetime'] = pd.to_datetime(sim_date + ' ' + dispatch_evts['Time'])

        # 找出每個波次的最後時間 (Max Time)
        actual_finish_times = dispatch_evts.groupby('WAVE_ID')['datetime'].max()

        # 3. 比對與計算延遲
        delays = []
        for wid, actual_time in actual_finish_times.items():
            if wid in wave_deadlines:
                deadline = wave_deadlines[wid]
                # 寬限期: 加上 30 分鐘作業時間 (假設最後一張單派出去還要 30 分鐘做完)
                estimated_completion = actual_time + timedelta(minutes=30)
                
                diff = (estimated_completion - deadline).total_seconds() / 60 # 分鐘
                
                status = "ON_TIME"
                if diff > 0: status = "DELAY"
                
                delays.append({
                    'WAVE_ID': wid,
                    'DEADLINE': deadline,
                    'LAST_DISPATCH': actual_time,
                    'EST_COMPLETION': estimated_completion,
                    'DELAY_MIN': round(diff, 1),
                    'STATUS': status
                })

        if not delays:
            print("   ⚠️ 無法計算延遲 (可能是 Log 無法解析 WAVE_ID)")
            return

        df_delay = pd.DataFrame(delays)
        avg_delay = df_delay[df_delay['STATUS']=='DELAY']['DELAY_MIN'].mean()
        max_delay = df_delay['DELAY_MIN'].max()
        delayed_waves = len(df_delay[df_delay['STATUS']=='DELAY'])
        
        print(f"   -> 總波次數: {len(df_delay)}")
        print(f"   -> 延遲波次數: {delayed_waves} ({delayed_waves/len(df_delay)*100:.1f}%)")
        print(f"   -> 平均延遲: {avg_delay:.1f} 分鐘")
        print(f"   -> 最大延遲: {max_delay:.1f} 分鐘")
        
        # 輸出 CSV
        out_path = os.path.join(LOG_DIR, 'wave_delay_report.csv')
        df_delay.to_csv(out_path, index=False)
        print(f"   ✅ 詳細延遲報告已輸出: {out_path}")

    # ==========================================
    # 驗證 3: 優先級與策略 (Priority & Strategy)
    # ==========================================
    def verify_logic(self):
        print("\n🧠 [3. 策略與優先級邏輯驗證]")
        if self.event_log.empty: return

        # 統計派單原因
        dispatch_reasons = self.event_log[self.event_log['Category']=='DISPATCH']['Details'].apply(
            lambda x: x.split('assigned ')[1].split(' ')[0] if 'assigned ' in x else 'UNK'
        )
        print("   -> 派單原因分佈:")
        print(dispatch_reasons.value_counts().to_string())
        
        # 檢查是否真的有「強制找新料架」
        inbound_new = len(self.event_log[
            (self.event_log['Category']=='STRATEGY') & 
            (self.event_log['Details'].str.contains('NEW_SLOT', na=False))
        ])
        inbound_total = len(self.event_log[self.event_log['Category']=='STRATEGY'])
        
        print(f"   -> 進貨策略: 共 {inbound_total} 次決策")
        # 這裡只能粗略看，因為我們在 Step 5 的 Log 格式比較簡單

        # 檢查資源鎖定頻率 (Locking)
        # 我們在 Step 5 沒有顯式 Log "LOCK"，但可以看 Dispatch 是否成功
        # 這裡從側面推敲：如果完工率高且沒有報錯，代表鎖定機制運作正常

if __name__ == "__main__":
    validator = SimulationValidator()
    validator.check_throughput()
    validator.analyze_wave_delays()
    validator.verify_logic()