import pandas as pd
import os
import re
from collections import defaultdict
import datetime

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
EVENTS_PATH = os.path.join(LOG_DIR, 'simulation_events.csv')
# ----------------------------------------

def load_events():
    if not os.path.exists(EVENTS_PATH):
        print(f"❌ File not found: {EVENTS_PATH}")
        return None
    try:
        df = pd.read_csv(EVENTS_PATH, on_bad_lines='skip', engine='python')
        df['start_ts'] = pd.to_datetime(df['start_time'], errors='coerce')
        df = df.dropna(subset=['start_ts']).sort_values('start_ts')
        return df
    except Exception as e:
        print(f"❌ Error reading csv: {e}")
        return None

def normalize_id(val):
    val = str(val).strip()
    if val.isdigit(): return f"AGV_{val}"
    if val.upper().startswith('AGV'): return f"AGV_{re.findall(r'\d+', val)[0]}"
    return val

def analyze_purple_zombies():
    print("🕵️‍♂️ Starting Diagnosis: The Case of the Purple Zombies...")
    df = load_events()
    if df is None or df.empty: return

    # State Tracking
    agv_state = defaultdict(lambda: {'loaded': False, 'task_type': 'IDLE', 'pos': None, 'last_move_ts': 0, 'shelf_id': None})
    
    # Time window for "Zombie" detection (10 mins)
    ZOMBIE_THRESHOLD_SEC = 600 
    
    # Sampling output every minute
    start_time = df['start_ts'].min()
    end_time = df['start_ts'].max()
    curr_sample_time = start_time.timestamp()
    
    print(f"⏱ Data Range: {start_time} to {end_time}")
    print("-" * 80)
    print(f"{'Time':<20} | {'Total Purple':<12} | {'Order Task':<10} | {'Rescue Task':<11} | {'Zombies (>10m)':<15}")
    print("-" * 80)

    events = df.to_dict('records')
    
    max_purple = 0
    
    for evt in events:
        ts = evt['start_ts'].timestamp()
        
        # --- Periodical Sampling Report ---
        while ts > curr_sample_time + 60:
            # Count current stats
            purple_count = 0
            order_purple = 0
            rescue_purple = 0
            zombie_count = 0
            
            for agv_id, state in agv_state.items():
                if state['loaded']:
                    purple_count += 1
                    if state['task_type'] == 'RESCUE': rescue_purple += 1
                    else: order_purple += 1
                    
                    if (curr_sample_time - state['last_move_ts']) > ZOMBIE_THRESHOLD_SEC:
                        zombie_count += 1
            
            if purple_count > max_purple: max_purple = purple_count
            
            t_str = datetime.datetime.fromtimestamp(curr_sample_time).strftime('%H:%M:%S')
            print(f"{t_str:<20} | {purple_count:<12} | {order_purple:<10} | {rescue_purple:<11} | {zombie_count:<15}")
            
            curr_sample_time += 60

        # --- Process Event ---
        aid = normalize_id(evt['obj_id'])
        if not aid.startswith('AGV'): continue
        
        etype = evt['type']
        
        # Track Position & Activity
        agv_state[aid]['last_move_ts'] = ts
        
        if etype == 'SHELF_LOAD':
            agv_state[aid]['loaded'] = True
            agv_state[aid]['shelf_id'] = evt['text']
            # If text is shelf ID only, it's likely an Order. 
            # Rescue moves usually logged as SHUFFLE_LOAD or we infer from context?
            # In Step4 code: Rescue uses SHUFFLE_LOAD. Normal uses SHELF_LOAD.
            agv_state[aid]['task_type'] = 'ORDER'
            agv_state[aid]['pos'] = (evt['sx'], evt['sy']) # Initially at shelf

        elif etype == 'SHUFFLE_LOAD':
            agv_state[aid]['loaded'] = True
            agv_state[aid]['shelf_id'] = evt['text']
            agv_state[aid]['task_type'] = 'RESCUE'
            agv_state[aid]['pos'] = (evt['sx'], evt['sy'])

        elif etype == 'SHELF_UNLOAD' or etype == 'SHUFFLE_UNLOAD':
            agv_state[aid]['loaded'] = False
            agv_state[aid]['task_type'] = 'IDLE'
            agv_state[aid]['shelf_id'] = None
            agv_state[aid]['pos'] = (evt['ex'], evt['ey'])

        elif etype in ['AGV_MOVE', 'FORCE_TELE']:
            agv_state[aid]['pos'] = (evt['ex'], evt['ey'])

    print("-" * 80)
    print(f"🚨 Max Purple AGVs Observed: {max_purple}")
    
    # Final Zombie Report
    print("\n🧟 Final Zombie AGV Report (Stuck at end of log):")
    stuck_agvs = []
    final_ts = end_time.timestamp()
    for agv_id, state in agv_state.items():
        if state['loaded'] and (final_ts - state['last_move_ts'] > ZOMBIE_THRESHOLD_SEC):
            duration_min = int((final_ts - state['last_move_ts']) / 60)
            stuck_agvs.append((agv_id, state['task_type'], duration_min, state['pos']))
    
    if not stuck_agvs:
        print("No zombies found at the very end.")
    else:
        stuck_agvs.sort(key=lambda x: x[2], reverse=True)
        for z in stuck_agvs:
            print(f"  - {z[0]} ({z[1]}): Stuck for {z[2]} mins at {z[3]}")
            
    print("\n💡 Analysis:")
    if max_purple > 32:
        print(f"  👉 The count ({max_purple}) exceeds the system limit of 32.")
        print("  👉 Check the 'Rescue Task' column above. High numbers indicate deadlock handling attempts that failed.")
    else:
        print("  👉 The count is within limits, but traffic might be stuck.")

if __name__ == "__main__":
    analyze_purple_zombies()