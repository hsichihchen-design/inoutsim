import pandas as pd
import json
import os
import re
import math

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_MAP_DIR = os.path.join(BASE_DIR, 'data', 'master')
MAPPING_DIR = os.path.join(BASE_DIR, 'data', 'mapping')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
OUTPUT_HTML = os.path.join(LOG_DIR, 'dashboard_report.html')
# ----------------------------------------

def load_map_fixed(filename, rows_limit, cols_limit):
    path = os.path.join(DATA_MAP_DIR, filename)
    df = None
    if os.path.exists(path):
        try: df = pd.read_excel(path, header=None)
        except: pass
    if df is None:
        csv_path = path.replace('.xlsx', '.csv')
        if os.path.exists(csv_path):
            try: df = pd.read_csv(csv_path, header=None)
            except: pass
    
    if df is not None:
        df = df.iloc[0:rows_limit, 0:cols_limit]
        grid = df.fillna(0).values.tolist()
        return grid
    return []

def load_shelf_map():
    path = os.path.join(MAPPING_DIR, 'shelf_coordinate_map.csv')
    shelf_set = {'2F': set(), '3F': set()}
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            for _, r in df.iterrows():
                if 0 <= r['x'] < 61 and 0 <= r['y'] < 32:
                    shelf_set[r['floor']].add((int(r['x']), int(r['y'])))
        except: pass
    return shelf_set

def normalize_obj_id(val):
    val = str(val).strip()
    # Normalize numbers or AGV_X
    if val.isdigit():
        return f"AGV_{int(val)}"
    if val.upper().startswith('AGV'):
        nums = re.findall(r'\d+', val)
        if nums:
            return f"AGV_{int(nums[0])}"
    return val

def generate_snapshots(events, initial_shelves, interval_seconds=300):
    """
    Python-side pre-calculation of state snapshots.
    To avoid huge file size, we only store the 'Diff': 
    Which shelves are currently NOT on the floor (i.e., loaded on AGVs).
    """
    print("📸 生成狀態快照 (Snapshots)...")
    
    # Track current state
    # current_missing_shelves: stores set of "x,y" strings for shelves currently picked up
    current_missing = {'2F': set(), '3F': set()} 
    
    snapshots = []
    last_snap_time = -interval_seconds
    
    # We need to simulate the state sequentially
    for idx, e in enumerate(events):
        start_ts = e[0] # start_time
        type_ = e[8]    # event type
        floor = e[2]
        
        # 1. Capture Snapshot if interval passed
        if start_ts - last_snap_time >= interval_seconds:
            snapshots.append({
                'time': start_ts,
                'index': idx,
                'missing_2F': list(current_missing['2F']),
                'missing_3F': list(current_missing['3F'])
            })
            last_snap_time = start_ts
            
        # 2. Update State
        # Load events (Shelf removed from floor)
        if type_ in ['SHELF_LOAD', 'SHUFFLE_LOAD']:
            key = f"{e[4]},{e[5]}" # sx, sy
            current_missing[floor].add(key)
            
        # Unload events (Shelf returned to floor)
        # Note: Unload happens at end coordinates (ex, ey)
        elif type_ in ['SHELF_UNLOAD', 'SHUFFLE_UNLOAD']:
            # SHUFFLE logic in step 4 moves things around.
            # The event is: start(t), end(t), floor, obj, sx, sy, ex, ey, TYPE...
            # When UNLOAD happens, the shelf appears at ex, ey.
            # However, our logic tracks "missing" by ID or original pos? 
            # Simplified: We just assume the shelf at destination is now occupied.
            # Wait, if we track by coordinate, SHUFFLE moves shelf from A to B.
            # LOAD A: A is missing. UNLOAD B: B is present.
            # This works perfectly for the visualizer which draws based on coordinates.
            
            # Key issue: If we SHUFFLE, we pick up at A, drop at B.
            # The visualizer draws shelves based on `initial_shelves` minus `current_missing` plus `moved_shelves`?
            # No, standard approach:
            # The map has static shelf locations.
            # If we move a shelf to a temp buffer, that buffer spot becomes "Shelf".
            # This is complex dynamic mapping. 
            
            # V50 Simplification for Speed:
            # We track `dynamic_shelves` (spots that have a shelf NOW).
            # Initial state: `dynamic_shelves` = `initial_shelves`
            # LOAD at (x,y): remove (x,y) from `dynamic_shelves`
            # UNLOAD at (x,y): add (x,y) to `dynamic_shelves`
            pass

    # Re-run logic to be cleaner for the JS side consumption
    # Actually, let's keep it simple. The JS will maintain a Set of valid shelf coordinates.
    # The snapshot will store the FULL LIST of valid shelf coordinates at that time?
    # No, that's too big (thousands of shelves).
    # Better: Store the operations delta? No, seeks need absolute state.
    
    # Optimal: Store the list of 'Modified Shelves' relative to initial map?
    # Let's go with: Snapshot stores the list of REMOVED shelves and ADDED shelves relative to base map.
    
    return [] # Placeholder, we will implement the logic inside JS for cleaner code distribution 
              # or do it here. Doing it here is better for performance.

def precompute_snapshots_robust(events, initial_shelf_sets):
    """
    Full simulation to generate lightweight snapshots.
    Snapshot Structure: { time: T, idx: I, current_shelves_2F: [...], current_shelves_3F: [...] }
    Wait, sending full array of 2000 shelves every 5 mins is heavy.
    
    Revised Strategy:
    The shelf layout changes (Shuffles).
    We will store:
    1. `removed`: Coordinates present in Initial but now empty.
    2. `added`: Coordinates not in Initial but now occupied (Buffer spots).
    """
    print("📸 預計算高速快照 (V50)...")
    
    current_shelves = {
        '2F': set(initial_shelf_sets['2F']),
        '3F': set(initial_shelf_sets['3F'])
    }
    
    # Convert sets to string keys "x,y" for easier diffing
    base_sets = {
        '2F': {f"{x},{y}" for x,y in initial_shelf_sets['2F']},
        '3F': {f"{x},{y}" for x,y in initial_shelf_sets['3F']}
    }
    curr_sets = {
        '2F': set(base_sets['2F']),
        '3F': set(base_sets['3F'])
    }
    
    snapshots = []
    last_snap_time = -1000
    
    for idx, e in enumerate(events):
        start_ts = e[0]
        end_ts = e[1]
        floor = e[2]
        type_ = e[8]
        
        # Save snapshot every 300 sim seconds
        if start_ts - last_snap_time >= 300:
            # Calculate Delta to save space
            # removed: in Base but not in Curr
            # added: in Curr but not in Base
            snap = {
                't': int(start_ts),
                'i': idx,
                'd': {} # data by floor
            }
            for f in ['2F', '3F']:
                removed = list(base_sets[f] - curr_sets[f])
                added = list(curr_sets[f] - base_sets[f])
                snap['d'][f] = {'r': removed, 'a': added}
            
            snapshots.append(snap)
            last_snap_time = start_ts
            
        # Execute Logic
        if type_ in ['SHELF_LOAD', 'SHUFFLE_LOAD']:
            key = f"{e[4]},{e[5]}"
            if key in curr_sets[floor]:
                curr_sets[floor].remove(key)
                
        elif type_ in ['SHELF_UNLOAD', 'SHUFFLE_UNLOAD']:
            key = f"{e[6]},{e[7]}" # Drop at End coords
            curr_sets[floor].add(key)
            
    print(f"   -> 建立 {len(snapshots)} 個快照點，優化隨機存取效能。")
    return snapshots

def main():
    print("🚀 [Step 5] 啟動視覺化 (V50: Real-Time Performance Optimized)...")

    map_2f = load_map_fixed('2F_map.xlsx', 32, 61)
    map_3f = load_map_fixed('3F_map.xlsx', 32, 61)
    shelf_data = load_shelf_map() # {'2F': set((x,y)), ...}
    
    events_path = os.path.join(LOG_DIR, 'simulation_events.csv')
    if not os.path.exists(events_path): 
        print("❌ 找不到 simulation_events.csv")
        return

    try:
        df_events = pd.read_csv(events_path, on_bad_lines='skip', engine='python')
    except Exception as e:
        print(f"❌ Error reading events: {e}")
        return
    
    # 1. 時間處理
    df_events['start_ts'] = pd.to_datetime(df_events['start_time'], errors='coerce')
    df_events['end_ts'] = pd.to_datetime(df_events['end_time'], errors='coerce')
    df_events = df_events.dropna(subset=['start_ts', 'end_ts'])
    df_events = df_events[df_events['start_ts'].dt.year > 2020]
    
    if df_events.empty: 
        print("⚠️ 無有效事件資料")
        return

    df_events['obj_id'] = df_events['obj_id'].apply(normalize_obj_id)

    df_events['start_ts'] = df_events['start_ts'].astype('int64') // 10**9
    df_events['end_ts'] = df_events['end_ts'].astype('int64') // 10**9
    df_events = df_events.sort_values('start_ts')
    df_events['text'] = df_events['text'].fillna('').astype(str)
    
    min_time = int(df_events['start_ts'].min())
    max_time = int(df_events['end_ts'].max())
    print(f"   📅 時間範圍: {pd.to_datetime(min_time, unit='s')} ~ {pd.to_datetime(max_time, unit='s')}")

    # Convert to list for JSON
    events_data = df_events[['start_ts', 'end_ts', 'floor', 'obj_id', 'sx', 'sy', 'ex', 'ey', 'type', 'text']].values.tolist()
    
    # Pre-calculate Snapshots
    snapshots_data = precompute_snapshots_robust(events_data, shelf_data)

    # Info extraction
    all_agvs = sorted(list(df_events[df_events['obj_id'].str.startswith('AGV')]['obj_id'].unique()))
    all_stations = df_events[df_events['obj_id'].str.startswith('WS_')]['obj_id'].unique().tolist()
    
    # KPI Processing
    kpi_path = os.path.join(LOG_DIR, 'simulation_kpi.csv')
    kpi_raw = []
    calc_wave_totals = {}
    calc_recv_totals = {}

    try:
        df_kpi = pd.read_csv(kpi_path, on_bad_lines='skip', engine='python')
        df_kpi['finish_ts'] = pd.to_datetime(df_kpi['finish_time'], errors='coerce')
        df_kpi = df_kpi.dropna(subset=['finish_ts'])
        
        df_kpi['date'] = df_kpi['finish_ts'].dt.strftime('%Y-%m-%d')
        df_kpi['finish_ts'] = df_kpi['finish_ts'].astype('int64') // 10**9
        df_kpi = df_kpi.sort_values('finish_ts')
        
        for _, row in df_kpi.iterrows():
            wid = str(row['wave_id'])
            # Fill totals if logic allows, otherwise use what Step 4 provided
            if row['type'] == 'RECEIVING':
                d = row['date']
                calc_recv_totals[d] = calc_recv_totals.get(d, 0) + 1
            else:
                # Step 4 now provides total_in_wave, we can use max of it
                val = int(row.get('total_in_wave', 0))
                if val > calc_wave_totals.get(wid, 0):
                    calc_wave_totals[wid] = val
                
        kpi_raw = df_kpi[['finish_ts', 'type', 'wave_id', 'is_delayed', 'date', 'workstation', 'total_in_wave', 'deadline_ts']].values.tolist()

    except Exception as e: 
        print(f"⚠️ KPI Error: {e}")

    html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Warehouse Monitor V50 (Performance)</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; margin: 0; display: flex; flex-direction: column; height: 100vh; overflow: hidden; background: #eef1f5; }
        .header { background: #fff; height: 40px; padding: 0 20px; display: flex; align-items: center; border-bottom: 1px solid #ddd; flex-shrink: 0; }
        .main { display: flex; flex: 1; overflow: hidden; }
        .map-section { flex: 3; display: flex; flex-direction: column; padding: 10px; gap: 10px; overflow: hidden; }
        .floor-container { flex: 1; background: #fff; border: 1px solid #ccc; position: relative; display: flex; flex-direction: column; overflow: hidden; }
        .floor-label { position: absolute; top: 5px; left: 5px; background: rgba(255,255,255,0.9); padding: 2px 6px; font-weight: bold; font-size: 12px; z-index: 10; border: 1px solid #999; }
        .canvas-wrap { flex: 1; width: 100%; height: 100%; position: relative; }
        canvas { display: block; width: 100%; height: 100%; }
        .dash-section { flex: 1; min-width: 400px; max-width: 480px; background: #fff; border-left: 1px solid #ccc; display: flex; flex-direction: column; }
        .dash-content { flex: 1; overflow-y: auto; padding: 10px; }
        .panel { margin-bottom: 10px; border: 1px solid #eee; padding: 8px; border-radius: 4px; background: #fafafa; }
        .panel h4 { margin: 0 0 8px 0; border-bottom: 2px solid #007bff; font-size: 14px; color: #333; }
        .station-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 5px; }
        .station-card { border: 1px solid #ddd; padding: 4px; font-size: 10px; text-align: center; background: #fff; border-radius: 3px; display: flex; flex-direction: column; justify-content: center; height: 35px; }
        .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 3px; }
        .wave-item { font-size:11px; margin-bottom:5px; background:#fff; padding:5px; border:1px solid #ddd; }
        .progress-bg { height:6px; background:#eee; margin-top:2px; border-radius:3px; overflow:hidden; }
        .progress-fill { height:100%; transition:width 0.3s; }
        .controls { padding: 10px; background: #fff; border-top: 1px solid #ddd; display: flex; gap: 10px; align-items: center; }
        .legend { display: flex; gap: 10px; font-size: 11px; margin-bottom: 5px; flex-wrap: wrap; }
        .box { width: 12px; height: 12px; margin-right: 3px; border: 1px solid #666; }
        .floor-subtitle { font-size: 12px; font-weight: bold; color: #555; margin: 5px 0 2px 0; border-bottom: 1px dashed #ccc; }
    </style>
</head>
<body>
    <div class="header">
        <h3>🏭 倉儲戰情室 (V50: Real-Time & High Perf)</h3>
        <div style="flex:1"></div>
        <span id="timeDisplay" style="font-weight: bold;">--</span>
    </div>
    <div class="main">
        <div class="map-section">
            <div class="legend">
                <div style="display:flex;align-items:center"><div class="box" style="background:#8d6e63"></div>料架</div>
                <div style="display:flex;align-items:center"><div class="box" style="border-radius:50%;background:#00e5ff;border:1px solid #000"></div>AGV(空)</div>
                <div style="display:flex;align-items:center"><div class="box" style="background:#d500f9;border:1px solid #fff"></div>AGV(載貨)</div>
                <div style="display:flex;align-items:center"><div class="box" style="background:orange"></div>讓路/阻塞</div>
                <div style="display:flex;align-items:center"><div class="box" style="background:red"></div>瞬移/警告</div>
            </div>
            <div class="floor-container">
                <div class="floor-label">2F Map</div>
                <div class="canvas-wrap"><canvas id="c2"></canvas></div>
            </div>
            <div class="floor-container">
                <div class="floor-label">3F Map</div>
                <div class="canvas-wrap"><canvas id="c3"></canvas></div>
            </div>
        </div>
        <div class="dash-section">
            <div class="dash-content">
                <div class="panel">
                    <h4>📡 工作站狀態</h4>
                    <div class="floor-subtitle">2F Stations</div>
                    <div id="st-list-2f" class="station-grid">Wait...</div>
                    <div class="floor-subtitle" style="margin-top:10px">3F Stations</div>
                    <div id="st-list-3f" class="station-grid">Wait...</div>
                </div>
                <div class="panel">
                    <h4>🌊 波次進度 (Outbound)</h4>
                    <div id="wave-list">Wait...</div>
                </div>
                <div class="panel">
                    <h4>🚛 進貨進度 (Inbound)</h4>
                    <div id="recv-list">Wait...</div>
                </div>
                <div class="panel">
                    <h4>📊 統計指標</h4>
                    <div>Active AGV: <span id="val-active">0</span></div>
                    <div>Done: <span id="val-done">0</span> | Delay: <span id="val-delay" style="color:red">0</span></div>
                </div>
            </div>
            <div class="controls">
                <button onclick="togglePlay()" id="playBtn">Play</button>
                <input type="range" id="slider" style="flex:1" oninput="isSeeking=true" onchange="isSeeking=false; onSeek(this.value)">
                <select id="speed">
                    <option value="5">5x Speed</option>
                    <option value="10" selected>10x Speed</option>
                    <option value="30">30x Speed</option>
                    <option value="60">1 min/s</option>
                    <option value="300">5 min/s</option>
                </select>
            </div>
        </div>
    </div>
<script>
    // --- DATA INJECTION ---
    const map2F = __MAP2F__;
    const map3F = __MAP3F__;
    const initialShelfData = __SHELF_DATA__; 
    const events = __EVENTS__;
    const snapshots = __SNAPSHOTS__;
    const kpiRaw = __KPI_RAW__;
    const agvIds = __AGV_IDS__;
    const stIds = __STATION_IDS__;
    const waveTotals = __WAVE_TOTALS__;
    const recvTotals = __RECV_TOTALS__;
    
    let minTime = Number(__MIN_TIME__);
    let maxTime = Number(__MAX_TIME__);
    
    // --- INITIALIZATION ---
    if (isNaN(minTime)) minTime = Math.floor(Date.now()/1000);
    if (isNaN(maxTime)) maxTime = minTime + 3600;
    
    document.getElementById('slider').min = minTime;
    document.getElementById('slider').max = maxTime;
    document.getElementById('slider').value = minTime;

    // Base Sets
    const baseShelves = { '2F': new Set(initialShelfData['2F']), '3F': new Set(initialShelfData['3F']) };
    // Current State (Mutable)
    let currentShelves = { '2F': new Set(baseShelves['2F']), '3F': new Set(baseShelves['3F']) };
    
    let agvState = {};
    agvIds.forEach(id => { 
        agvState[id] = { floor: '2F', x: -1, y: -1, visible: false, color: '#00e5ff', loaded: false }; 
    });
    
    let stState = {};
    stIds.forEach(id => { 
        // ID format usually WS_2F_1
        let label = id;
        try {
            const parts = id.split('_');
            if (parts.length >= 3) label = parts[2]; // Get '1' from WS_2F_1
        } catch(e){}
        
        const f = id.includes('3F') ? '3F' : '2F';
        stState[id] = { status: 'IDLE', color: 'WHITE', floor: f, x:-1, y:-1, label: label, text: 'Idle' }; 
    });

    // --- CANVAS SETUP ---
    function setupCanvas(id, mapData) {
        const c = document.getElementById(id);
        const ctx = c.getContext('2d');
        const parent = c.parentElement;
        c.width = parent.clientWidth || 800; c.height = parent.clientHeight || 400;
        const rows = mapData.length || 10;
        const cols = mapData[0]?.length || 10;
        const scaleX = c.width / cols;
        const scaleY = c.height / rows;
        const size = Math.min(scaleX, scaleY);
        const ox = (c.width - cols*size)/2;
        const oy = (c.height - rows*size)/2;
        return { ctx, rows, cols, size, ox, oy, map: mapData };
    }
    
    let f2 = setupCanvas('c2', map2F);
    let f3 = setupCanvas('c3', map3F);
    window.onresize = () => { f2 = setupCanvas('c2', map2F); f3 = setupCanvas('c3', map3F); render(); };

    // --- BINARY SEARCH UTILS ---
    function bisectRight(arr, t) {
        let lo = 0, hi = arr.length;
        while (lo < hi) {
            let mid = (lo + hi) >>> 1;
            if (arr[mid][0] <= t) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }
    
    function bisectSnapshots(arr, t) {
        let lo = 0, hi = arr.length;
        while (lo < hi) {
            let mid = (lo + hi) >>> 1;
            if (arr[mid].t <= t) lo = mid + 1;
            else hi = mid;
        }
        return lo - 1;
    }

    // --- PLAYBACK STATE ---
    let currTime = minTime;
    let lastProcessedIdx = 0;
    let isPlaying = false;
    let isSeeking = false;

    // --- CORE LOGIC ---
    function restoreSnapshot(snapIdx) {
        const snap = snapshots[snapIdx];
        if (!snap) return 0;
        
        // Restore Shelves using delta
        ['2F', '3F'].forEach(f => {
            currentShelves[f] = new Set(baseShelves[f]); // Reset to Base
            // Apply Removed
            snap.d[f].r.forEach(k => currentShelves[f].delete(k));
            // Apply Added
            snap.d[f].a.forEach(k => currentShelves[f].add(k));
        });
        
        // Reset AGVs (snapshot logic for AGVs is complex, so we just clear visuals and let events fill in)
        // Since we replay from snapshot time, AGVs will jump to position on next event.
        // For smoother visual, we could store AGV pos in snapshot, but strictly not required if events are frequent.
        agvIds.forEach(id => { 
            agvState[id].visible = false; 
        });

        return snap.i; // Return the event index to start processing from
    }

    function onSeek(val) {
        const targetTime = Number(val);
        currTime = targetTime;
        
        // 1. Find closest previous snapshot
        const snapIdx = bisectSnapshots(snapshots, targetTime);
        let startIdx = 0;
        
        if (snapIdx >= 0) {
            startIdx = restoreSnapshot(snapIdx);
        } else {
            // No snapshot before this time (early game), reset to zero
            ['2F', '3F'].forEach(f => currentShelves[f] = new Set(baseShelves[f]));
            startIdx = 0;
        }
        
        // 2. Replay events from Snapshot Time to Target Time
        // This 'Fast Forward' updates the state without rendering
        fastProcess(startIdx, targetTime);
        lastProcessedIdx = bisectRight(events, targetTime);
        render();
    }

    function fastProcess(startIdx, targetTime) {
        for(let i = startIdx; i < events.length; i++) {
            const e = events[i];
            if (e[0] > targetTime) break; // Past target
            processEventLogic(e, targetTime, false);
        }
    }

    function processEventLogic(e, time, isRealtime) {
        const type = e[8];
        const floor = e[2];
        const startT = e[0];
        const endT = e[1];
        
        // --- SHELF LOGIC ---
        // 'SHUFFLE_LOAD' acts just like 'SHELF_LOAD' (Remove shelf from floor)
        if (type === 'SHELF_LOAD' || type === 'SHUFFLE_LOAD') {
            const key = e[4] + "," + e[5];
            currentShelves[floor].delete(key);
            if (e[3].startsWith('AGV')) agvState[e[3]].loaded = true;
        } 
        // 'SHUFFLE_UNLOAD' acts just like 'SHELF_UNLOAD' (Add shelf to floor)
        else if (type === 'SHELF_UNLOAD' || type === 'SHUFFLE_UNLOAD') {
            const key = e[6] + "," + e[7]; // Destination
            currentShelves[floor].add(key);
            if (e[3].startsWith('AGV')) agvState[e[3]].loaded = false;
        }

        // --- AGV MOVEMENT ---
        if (e[3].startsWith('AGV')) {
            const id = e[3];
            if (time >= startT) {
                agvState[id].floor = floor;
                agvState[id].visible = true;
                
                // Interpolation for smooth movement
                if (isRealtime && time <= endT && endT > startT) {
                    const p = (time - startT) / (endT - startT);
                    agvState[id].x = e[4] + (e[6] - e[4]) * p;
                    agvState[id].y = e[5] + (e[7] - e[5]) * p;
                } else {
                    // Static position (finished move)
                    agvState[id].x = e[6];
                    agvState[id].y = e[7];
                }
                
                // Color Logic
                if (type === 'YIELD') agvState[id].color = 'orange';
                else if (type === 'PARKING') agvState[id].color = 'green';
                else if (type.includes('TELE') || type === 'FORCE_TELE') agvState[id].color = 'red';
                else if (type.includes('SHUFFLE')) agvState[id].color = '#aa00ff'; // Darker purple for shuffle
                else {
                    agvState[id].color = agvState[id].loaded ? '#d500f9' : '#00e5ff';
                }
            }
        }
        
        // --- STATION STATUS ---
        if (type === 'STATION_STATUS') {
            // "BLUE|Station 1|N"
            if (time >= startT && time < endT) {
                const sid = e[3];
                if (stState[sid]) {
                    const parts = e[9].split('|');
                    stState[sid].color = parts[0];
                    stState[sid].text = parts[1] || 'Busy';
                }
            }
        }
    }

    function updateStateRealtime(time) {
        // Optimized: Only process events relevant to current frame window or catch up
        // Find where we should be
        const endIdx = bisectRight(events, time);
        
        // If we jumped too far back or forward without seek, handle it
        if (Math.abs(endIdx - lastProcessedIdx) > 500) {
            onSeek(time); // Fallback to seek logic for safety
            return;
        }

        // Process strictly forward
        for(let i = lastProcessedIdx; i < endIdx; i++) {
            processEventLogic(events[i], time, true);
        }
        
        // Also update interpolation for currently active events (events that span across 'time')
        // We can scan backward a bit or keep a set of 'active' events. 
        // Simple hack: Scan last 50 events for any that encompass `time`.
        const scanStart = Math.max(0, endIdx - 50);
        for(let i = scanStart; i < endIdx; i++) {
            const e = events[i];
            if (e[1] > time) { // It's still finishing
                processEventLogic(e, time, true);
            }
        }

        lastProcessedIdx = endIdx;
    }

    // --- DRAWING ---
    function drawMap(obj, floorName) {
        const ctx = obj.ctx;
        ctx.fillStyle = '#fafafa'; ctx.fillRect(0,0, ctx.canvas.width, ctx.canvas.height);
        if(!obj.map) return;
        
        // Draw Grid
        for(let r=0; r<obj.rows; r++) {
            for(let c=0; c<obj.cols; c++) {
                const val = obj.map[r][c];
                const x = obj.ox + c * obj.size;
                const y = obj.oy + r * obj.size;
                const s = obj.size;
                
                // -1: Wall, 0: Aisle, 1: Storage (but might be empty), 2: Station
                if (val === -1) { ctx.fillStyle = '#ccc'; ctx.fillRect(x,y,s,s); }
                else if (val === 2) { ctx.strokeStyle = '#bbb'; ctx.strokeRect(x,y,s,s); }
                else { 
                    // Floor
                    ctx.fillStyle = 'white'; ctx.fillRect(x,y,s,s); 
                    // Shelf check
                    const key = c + "," + r;
                    if (currentShelves[floorName].has(key)) {
                        ctx.fillStyle = '#8d6e63';
                        ctx.fillRect(x+1,y+1,s-2,s-2);
                    }
                }
            }
        }
        
        // Draw Stations
        Object.keys(stState).forEach(sid => {
            const s = stState[sid];
            if(s.floor === floorName && s.x !== -1) {
                const x=obj.ox+s.x*obj.size, y=obj.oy+s.y*obj.size, sz=obj.size;
                ctx.fillStyle = s.color === 'BLUE' ? '#007bff' : s.color === 'GREEN' ? '#28a745' : '#ddd';
                ctx.fillRect(x+1, y+1, sz-2, sz-2);
                ctx.fillStyle = 'black'; ctx.font = 'bold 8px Arial';
                ctx.fillText(s.label, x+2, y+sz/1.5);
            }
        });
    }

    function render() {
        if (!isSeeking) updateStateRealtime(currTime);
        
        drawMap(f2, '2F');
        drawMap(f3, '3F');
        
        // Draw AGVs
        let activeCount = 0;
        Object.keys(agvState).forEach(id => {
            const s = agvState[id];
            if (!s.visible) return;
            activeCount++;
            const obj = s.floor == '2F' ? f2 : f3;
            if(!obj.map) return;
            const sz = obj.size;
            const px = obj.ox + s.x * sz + sz/2;
            const py = obj.oy + s.y * sz + sz/2;
            
            obj.ctx.fillStyle = s.color;
            obj.ctx.beginPath();
            obj.ctx.arc(px, py, sz/2.1, 0, Math.PI*2);
            obj.ctx.fill();
            obj.ctx.strokeStyle = '#333'; obj.ctx.lineWidth = 1; obj.ctx.stroke();
            
            // Draw box if loaded
            if (s.loaded && !['red','orange','green'].includes(s.color)) {
                obj.ctx.fillStyle = '#fff';
                obj.ctx.fillRect(px-sz/4, py-sz/4, sz/2, sz/2);
            }
            
            // Label (Just Number)
            if (sz > 8) {
                obj.ctx.fillStyle = 'black'; 
                obj.ctx.font = 'bold 9px Arial';
                obj.ctx.textAlign = 'center';
                const label = id.replace('AGV_', '');
                obj.ctx.fillText(label, px, py+3);
            }
        });
        document.getElementById('val-active').innerText = activeCount;

        // KPI & Time
        const dObj = new Date(currTime*1000);
        document.getElementById('timeDisplay').innerText = dObj.toLocaleString();
        if (!isSeeking) document.getElementById('slider').value = currTime;

        // Progress Lists (Simplified for perf)
        // Only update these once per second
        if (Math.floor(currTime) % 2 === 0) updateDashboardLists(dObj);
    }

    function updateDashboardLists(dObj) {
        // Find completed tasks up to currTime
        // KPI is pre-sorted by finish_time
        const kIdx = bisectRight(kpiRaw, currTime);
        const doneSlice = kpiRaw.slice(0, kIdx);
        
        document.getElementById('val-done').innerText = doneSlice.length;
        
        let delayed = 0;
        let waveProg = {};
        let recvProg = {};
        
        doneSlice.forEach(k => {
            if (k[3] === 'Y') delayed++;
            if (k[1] === 'RECEIVING') recvProg[k[4]] = (recvProg[k[4]]||0) + 1;
            else waveProg[k[2]] = (waveProg[k[2]]||0) + 1;
        });
        document.getElementById('val-delay').innerText = delayed;

        // Wave HTML
        let wHtml = '';
        const activeWaves = Object.keys(waveTotals).sort(); 
        activeWaves.forEach(wid => {
            const total = waveTotals[wid];
            const done = waveProg[wid] || 0;
            // Show if active or recently finished
            if (done < total || (done >= total && Math.random() > 0.9)) { 
                const pct = total > 0 ? Math.min(100, (done/total*100)).toFixed(0) : 0;
                wHtml += `<div class="wave-item"><div style="display:flex;justify-content:space-between"><span>${wid}</span><span>${done}/${total}</span></div><div class="progress-bg"><div class="progress-fill" style="width:${pct}%;background:#007bff"></div></div></div>`;
            }
        });
        document.getElementById('wave-list').innerHTML = wHtml || '<div style="color:#999;padding:5px">No Active Waves</div>';
        
        // Station HTML
        let h2 = '', h3 = '';
        Object.keys(stState).forEach(sid => {
            const s = stState[sid];
            const card = `<div class="station-card"><div style="font-weight:bold">${s.label}</div><div style="margin-top:2px"><span class="status-dot" style="background:${s.color === 'WHITE' ? '#ddd' : s.color === 'BLUE' ? '#007bff' : '#28a745'}"></span>${s.text}</div></div>`;
            if (s.floor === '2F') h2 += card; else h3 += card;
        });
        document.getElementById('st-list-2f').innerHTML = h2;
        document.getElementById('st-list-3f').innerHTML = h3;
    }

    // --- ANIMATION LOOP ---
    function animate() {
        if(isPlaying && !isSeeking) {
            const speed = parseInt(document.getElementById('speed').value);
            currTime += (1/30) * speed; 
            if(currTime > maxTime) { isPlaying=false; currTime=minTime; onSeek(minTime); }
            render();
        }
        requestAnimationFrame(animate);
    }
    
    function togglePlay() { isPlaying=!isPlaying; }
    
    // Initial Render
    onSeek(minTime);
    animate();
    
</script>
</body>
</html>
"""
    
    # Serialize Shelf Data correctly
    js_shelf_data = {'2F': [], '3F': []}
    for f in shelf_data: js_shelf_data[f] = [f"{c[0]},{c[1]}" for c in shelf_data[f]]

    final_html = html_template.replace('__MAP2F__', json.dumps(map_2f)) \
                              .replace('__MAP3F__', json.dumps(map_3f)) \
                              .replace('__SHELF_DATA__', json.dumps(js_shelf_data)) \
                              .replace('__EVENTS__', json.dumps(events_data)) \
                              .replace('__SNAPSHOTS__', json.dumps(snapshots_data)) \
                              .replace('__KPI_RAW__', json.dumps(kpi_raw)) \
                              .replace('__AGV_IDS__', json.dumps(all_agvs)) \
                              .replace('__STATION_IDS__', json.dumps(all_stations)) \
                              .replace('__MIN_TIME__', str(min_time)) \
                              .replace('__MAX_TIME__', str(max_time)) \
                              .replace('__WAVE_TOTALS__', json.dumps(calc_wave_totals)) \
                              .replace('__RECV_TOTALS__', json.dumps(calc_recv_totals))

    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(final_html)
    print(f"✅ 視覺化生成完畢: {OUTPUT_HTML} (V50: High Performance)")

if __name__ == "__main__":
    main()