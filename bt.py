import pandas as pd
import numpy as np
import requests
import time
import threading
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime, timedelta
from collections import deque

# --- CONFIGURATION ---
SYMBOL = "ETHUSDT"
HISTORY_DAYS = 5
CSV_PATH = '/app/data/ethohlc1m.csv'
HTTP_PORT = 8080

# --- SHARED STATE ---
# This variable is updated by the predictor loop and read by the HTTP server
LATEST_METRICS = {
    "netSum": 0,
    "activePredictions": 0,
    "lastUpdate": None
}

# --- 1. HTTP SERVER ---

class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Respond to any GET request with the JSON metrics
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        # Create a clean copy to avoid race conditions (minimal risk here but good practice)
        response_data = LATEST_METRICS.copy()
        self.wfile.write(json.dumps(response_data).encode('utf-8'))

    def log_message(self, format, *args):
        # Silence default server logs to keep console clean for predictor output
        pass

def start_server():
    server_address = ('0.0.0.0', HTTP_PORT)
    httpd = HTTPServer(server_address, MetricsHandler)
    print(f"\n[SERVER] Serving metrics at http://localhost:{HTTP_PORT}")
    httpd.serve_forever()

# --- 2. DATA & MODEL UTILS ---

def load_local_data(filepath):
    print(f"Loading local data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        df.columns = [c.lower() for c in df.columns]
        rename_map = {c: 'open' for c in df.columns if 'open' in c}
        rename_map.update({c: 'high' for c in df.columns if 'high' in c})
        rename_map.update({c: 'low' for c in df.columns if 'low' in c})
        rename_map.update({c: 'close' for c in df.columns if 'close' in c})
        rename_map.update({c: 'datetime' for c in df.columns if 'date' in c or 'time' in c})
        df.rename(columns=rename_map, inplace=True)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        return df.sort_index()
    except FileNotFoundError:
        print("CSV not found.")
        return pd.DataFrame()

def process_candles(df, window=1440):
    daily = pd.DataFrame(index=df.index)
    daily['close'] = df['close']
    daily['open'] = df['open'].shift(window - 1)
    daily['high'] = df['high'].rolling(window=window).max()
    daily['low'] = df['low'].rolling(window=window).min()
    daily.dropna(inplace=True)
    
    shift = daily['open'] - 1
    norm_high = daily['high'] - shift
    norm_low = daily['low'] - shift
    norm_close = daily['close'] - shift
    
    rng = (norm_high - norm_low).replace(0, 1e-9)
    section_size = rng / 8
    bins = ((norm_close - norm_low) / section_size).astype(int).clip(0, 7)
    
    daily['state'] = bins
    daily['norm_low'] = norm_low
    daily['section_size'] = section_size
    daily['s1'] = daily['state'].shift(2880)
    daily['s2'] = daily['state'].shift(1440)
    
    return daily

def train_model(train_df):
    if train_df.empty: return {}
    data = train_df[['s1', 's2', 'state']].dropna()
    counts = data.groupby(['s1', 's2', 'state']).size().reset_index(name='count')
    best = counts.sort_values('count', ascending=False).drop_duplicates(subset=['s1', 's2'])
    return {(int(row['s1']), int(row['s2'])): int(row['state']) for _, row in best.iterrows()}

def fetch_recent_history(symbol, days):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    base_url = "https://api.binance.com/api/v3/klines"
    
    all_data = []
    current_start = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)
    
    print(f"Fetching history since {start_time}...")
    while current_start < end_ts:
        params = {'symbol': symbol, 'interval': '1m', 'startTime': current_start, 'limit': 1000}
        try:
            r = requests.get(base_url, params=params)
            data = r.json()
            if not data: break
            all_data.extend(data)
            current_start = data[-1][0] + 60000
            time.sleep(0.05)
        except Exception as e:
            print(f"Fetch error: {e}")
            break
            
    df = pd.DataFrame(all_data, columns=['open_time', 'open', 'high', 'low', 'close', 'v', 'ct', 'qv', 'n', 'tb', 'tq', 'i'])
    df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
    for c in ['open', 'high', 'low', 'close']: df[c] = df[c].astype(float)
    df.set_index('datetime', inplace=True)
    return df[['open', 'high', 'low', 'close']].sort_index()

# --- 3. LIVE PREDICTOR ---

class LiveUniversePredictor:
    def __init__(self, model_map):
        self.model = model_map
        self.raw_history = pd.DataFrame() 
        self.active_predictions = deque() 
        
    def update_data(self, new_df):
        self.raw_history = pd.concat([self.raw_history, new_df])
        self.raw_history = self.raw_history[~self.raw_history.index.duplicated(keep='last')]
        if len(self.raw_history) > 5000:
            self.raw_history = self.raw_history.iloc[-5000:]
            
    def predict_next(self):
        processed = process_candles(self.raw_history)
        if processed.empty: return

        current_row = processed.iloc[-1]
        
        # Check history availability
        if pd.isna(current_row['s2']) or pd.isna(current_row['state']):
            return
            
        input_s1 = int(current_row['s2'])
        input_s2 = int(current_row['state'])
        
        key = (input_s1, input_s2)
        
        if key in self.model:
            pred_bin = self.model[key]
            
            # Open (T+1440) = Close (T). Normalized Open = 1.
            # Predict if bin center > 1
            pred_center_norm = current_row['norm_low'] + (pred_bin + 0.5) * current_row['section_size']
            direction = 1 if pred_center_norm > 1.0 else -1
            
            expiry = current_row.name + timedelta(minutes=1440)
            self.active_predictions.append({
                'created_at': current_row.name,
                'expiry': expiry,
                'direction': direction
            })
            
            print(f"[{current_row.name}] Seq({input_s1}, {input_s2}) -> Pred Bin {pred_bin} (Dir: {direction})")

    def prune_predictions(self):
        now = self.raw_history.index[-1]
        while self.active_predictions and self.active_predictions[0]['expiry'] <= now:
            self.active_predictions.popleft()
            
    def get_net_exposure(self):
        return sum(p['direction'] for p in self.active_predictions)

# --- 4. MAIN EXECUTION ---

if __name__ == "__main__":
    # A. Train
    print("--- 1. Training Model ---")
    raw_train = load_local_data(CSV_PATH)
    processed_train = process_candles(raw_train)
    prob_map = train_model(processed_train)
    print(f"Model Trained. Patterns: {len(prob_map)}")

    # B. Start Server Thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # C. Initialize Predictor
    print("\n--- 2. Initializing Live History ---")
    predictor = LiveUniversePredictor(prob_map)
    init_df = fetch_recent_history(SYMBOL, HISTORY_DAYS)
    predictor.update_data(init_df)
    
    # D. Prediction Loop
    print("\n--- 3. Starting Live Prediction Loop ---")
    print("Waiting for synchronization (00s)...")
    
    while True:
        now = datetime.now()
        sleep_sec = 60 - now.second + 1
        if sleep_sec > 60: sleep_sec = 1
        
        print(f"Sleeping {sleep_sec}s...")
        time.sleep(sleep_sec)
        
        try:
            # Fetch latest candle
            r = requests.get("https://api.binance.com/api/v3/klines", 
                             params={'symbol': SYMBOL, 'interval': '1m', 'limit': 5})
            data = r.json()
            
            # Format
            cols = ['open_time', 'open', 'high', 'low', 'close', 'v', 'ct', 'qv', 'n', 'tb', 'tq', 'i']
            new_df = pd.DataFrame(data, columns=cols)
            new_df['datetime'] = pd.to_datetime(new_df['open_time'], unit='ms')
            for c in ['open', 'high', 'low', 'close']: new_df[c] = new_df[c].astype(float)
            new_df.set_index('datetime', inplace=True)
            
            # Process
            predictor.update_data(new_df[['open', 'high', 'low', 'close']])
            predictor.prune_predictions()
            predictor.predict_next()
            
            # Update Shared State
            net_exp = predictor.get_net_exposure()
            LATEST_METRICS["netSum"] = net_exp
            LATEST_METRICS["activePredictions"] = len(predictor.active_predictions)
            LATEST_METRICS["lastUpdate"] = str(datetime.now())
            
            print(f"Active: {len(predictor.active_predictions)} | Net Sum: {net_exp:+d}")
            print("-" * 40)
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)
