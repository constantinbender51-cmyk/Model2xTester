import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from collections import deque

# --- CONFIGURATION ---
SYMBOL = "ETHUSDT"
HISTORY_DAYS = 5       # Need enough for S1 (2 days ago) + S2 (1 day ago)
WINDOW = 1440          # 1 Day in minutes
CSV_PATH = '/app/data/ethohlc1m.csv'

# --- 1. REUSED UTILS (From Previous Steps) ---
def load_local_data(filepath):
    # (Same as before - simplified for brevity)
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

def process_candles(df, window=1440):
    # (Same as before - crucial for state calculation)
    daily = pd.DataFrame(index=df.index)
    daily['close'] = df['close']
    daily['open'] = df['open'].shift(window - 1)
    daily['high'] = df['high'].rolling(window=window).max()
    daily['low'] = df['low'].rolling(window=window).min()
    daily.dropna(inplace=True)
    
    # Normalize & Discretize
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
    
    # Lagged Features
    daily['s1'] = daily['state'].shift(2880)
    daily['s2'] = daily['state'].shift(1440)
    
    return daily

def train_model(train_df):
    data = train_df[['s1', 's2', 'state']].dropna()
    counts = data.groupby(['s1', 's2', 'state']).size().reset_index(name='count')
    best = counts.sort_values('count', ascending=False).drop_duplicates(subset=['s1', 's2'])
    return {(int(row['s1']), int(row['s2'])): int(row['state']) for _, row in best.iterrows()}

def fetch_recent_history(symbol, days):
    """Fetches enough history to initialize the rolling windows."""
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

# --- 2. LIVE PREDICTOR CLASS ---

class LiveUniversePredictor:
    def __init__(self, model_map):
        self.model = model_map
        # We keep track of the last N minutes of raw data to calculate states on the fly
        self.raw_history = pd.DataFrame() 
        # Queue to hold active predictions: (expire_time, direction, predicted_bin)
        self.active_predictions = deque() 
        
    def update_data(self, new_df):
        """Append new data and maintain buffer size."""
        # We need at least 3 days (approx 4320 mins) to calculate S1 (2880 ago)
        self.raw_history = pd.concat([self.raw_history, new_df])
        self.raw_history = self.raw_history[~self.raw_history.index.duplicated(keep='last')]
        
        # Keep buffer manageable (e.g., last 5000 minutes)
        if len(self.raw_history) > 5000:
            self.raw_history = self.raw_history.iloc[-5000:]
            
    def predict_next(self):
        """
        Calculates the state of the universe that just closed (Time T).
        Predicts the state for Time T + 1440.
        """
        # Recalculate indicators on the updated history
        # (Optimized: In production, we would update incrementally, but pandas is fast enough for 5k rows)
        processed = process_candles(self.raw_history)
        
        # Get the very last row (Time T)
        current_row = processed.iloc[-1]
        
        # We need S1 (T-2880) and S2 (T-1440) relative to T
        # Wait, the `process_candles` function shifts S1 by 2880 and S2 by 1440.
        # So `current_row['s1']` IS the state from 2 days ago, and `current_row['s2']` IS the state from 1 day ago.
        # `current_row['state']` is the state that JUST FINISHED (Time T).
        
        # Our Sequence Logic from Training:
        # Input: S1, S2 -> Target: S3 (Current State)
        # To Predict Future (T+1440):
        # We need Input: S_prev (T-1440), S_curr (T).
        # These correspond to `current_row['s2']` and `current_row['state']`.
        
        if pd.isna(current_row['s2']) or pd.isna(current_row['state']):
            print("Insufficient history to form a sequence.")
            return
            
        input_s1 = int(current_row['s2']) # The state 1440m ago
        input_s2 = int(current_row['state']) # The state now
        
        key = (input_s1, input_s2)
        
        if key in self.model:
            pred_bin = self.model[key]
            
            # Interpret Direction
            # Open of T+1440 will be Close of T (current_row['close'])
            # Normalized Open = 1.
            # We predict the shape of T+1440 relative to T+1440's High/Low.
            # Using current Volatility (section_size) as a proxy for future range properties
            
            pred_center_norm = current_row['norm_low'] + (pred_bin + 0.5) * current_row['section_size']
            
            # If predicted shape > 1 (Open), Direction is UP
            direction = 1 if pred_center_norm > 1.0 else -1
            
            # Store Prediction
            expiry = current_row.name + timedelta(minutes=1440)
            self.active_predictions.append({
                'created_at': current_row.name,
                'expiry': expiry,
                'direction': direction,
                'bin': pred_bin
            })
            
            print(f"[{current_row.name}] Seq({input_s1}, {input_s2}) -> Pred Bin {pred_bin} (Dir: {direction})")
        else:
            print(f"[{current_row.name}] Seq({input_s1}, {input_s2}) -> Unknown Pattern")

    def prune_predictions(self):
        """Removes predictions that have expired (older than 1440 mins)."""
        now = self.raw_history.index[-1]
        
        # In a real system, we would calculate PnL here by comparing with current price
        
        while self.active_predictions and self.active_predictions[0]['expiry'] <= now:
            expired = self.active_predictions.popleft()
            # (Optional) Calculate realized PnL of this expired prediction here
            
    def get_net_exposure(self):
        return sum(p['direction'] for p in self.active_predictions)

# --- 3. MAIN EXECUTION LOOP ---

if __name__ == "__main__":
    # A. Train Model
    print("--- 1. Training Model ---")
    try:
        raw_train = load_local_data(CSV_PATH)
        processed_train = process_candles(raw_train)
        prob_map = train_model(processed_train)
        print(f"Model Trained. Patterns: {len(prob_map)}")
    except Exception as e:
        print(f"Training Failed: {e}")
        exit()

    # B. Initialize Live System
    print("\n--- 2. Initializing Live History ---")
    predictor = LiveUniversePredictor(prob_map)
    
    # Load recent history from Binance to "warm up" the state
    init_df = fetch_recent_history(SYMBOL, HISTORY_DAYS)
    predictor.update_data(init_df)
    print(f"History loaded: {len(predictor.raw_history)} candles.")

    # C. Real-Time Loop
    print("\n--- 3. Starting Live Prediction Loop ---")
    print("Waiting for synchronization (00s)...")
    
    while True:
        now = datetime.now()
        
        # Sync to XX:XX:01
        # Calculate seconds to sleep to reach the next minute's 01s
        sleep_sec = 60 - now.second + 1
        if sleep_sec > 60: sleep_sec = 1
        
        print(f"Sleeping {sleep_sec}s until next candle...")
        time.sleep(sleep_sec)
        
        # 1. Fetch Latest 1m Candle
        # We fetch the last 2 minutes to be safe, but we only need the most recently closed one
        try:
            # Quick fetch of last 5 candles
            r = requests.get("https://api.binance.com/api/v3/klines", 
                             params={'symbol': SYMBOL, 'interval': '1m', 'limit': 5})
            data = r.json()
            
            # Convert to DF
            cols = ['open_time', 'open', 'high', 'low', 'close', 'v', 'ct', 'qv', 'n', 'tb', 'tq', 'i']
            new_df = pd.DataFrame(data, columns=cols)
            new_df['datetime'] = pd.to_datetime(new_df['open_time'], unit='ms')
            for c in ['open', 'high', 'low', 'close']: new_df[c] = new_df[c].astype(float)
            new_df.set_index('datetime', inplace=True)
            
            # Only keep finalized candles (ignore the currently forming one if returned)
            # Binance 'close_time' is useful here, or we just trust the timestamp
            # We want the candle that JUST closed.
            # If now is 12:05:01, we want the 12:04 candle (which closed at 12:04:59).
            # The API returns candles by Open Time. The 12:04 candle has Open Time 12:04.
            
            # Update predictor
            predictor.update_data(new_df[['open', 'high', 'low', 'close']])
            
            # 2. Prune Old Predictions
            predictor.prune_predictions()
            
            # 3. Predict New Universe
            predictor.predict_next()
            
            # 4. Print Sum
            net_exp = predictor.get_net_exposure()
            count = len(predictor.active_predictions)
            print(f"Active Predictions: {count}/1440 | Net Sum (Direction): {net_exp:+d}")
            print("-" * 40)
            
        except Exception as e:
            print(f"Error in loop: {e}")
            time.sleep(10) # Error backoff
