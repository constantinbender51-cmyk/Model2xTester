import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta

# --- 1. Data Loading & Fetching ---

def load_local_data(filepath):
    print(f"Loading local data from {filepath}...")
    df = pd.read_csv(filepath)
    # Standardize columns
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
    df.sort_index(inplace=True)
    return df

def fetch_binance_data(symbol="ETHUSDT", interval="1m", days=7):
    """
    Fetches the last N days of 1m kline data from Binance Public API.
    """
    print(f"Fetching last {days} days of {interval} data for {symbol} from Binance...")
    
    # Binance API constants
    base_url = "https://api.binance.com/api/v3/klines"
    limit = 1000  # Max per request
    
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    
    # Convert to milliseconds
    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)
    
    all_data = []
    current_start = start_ts
    
    while current_start < end_ts:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'endTime': end_ts,
            'limit': limit
        }
        
        try:
            r = requests.get(base_url, params=params)
            r.raise_for_status()
            data = r.json()
            
            if not data:
                break
                
            all_data.extend(data)
            
            # Update start time to the last candle's close time + 1ms
            # data format: [open_time, open, high, low, close, ...]
            last_open_time = data[-1][0]
            current_start = last_open_time + 60000 # Advance 1 minute
            
            # Rate limit politeness
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
    # Convert to DataFrame
    # Columns: Open Time, Open, High, Low, Close, Volume, Close Time, ...
    cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'q_vol', 'trades', 'tb_base', 'tb_quote', 'ignore']
    df = pd.DataFrame(all_data, columns=cols)
    
    # Type conversion
    df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
    for c in ['open', 'high', 'low', 'close']:
        df[c] = df[c].astype(float)
        
    df.set_index('datetime', inplace=True)
    df = df[['open', 'high', 'low', 'close']].sort_index()
    
    print(f"Fetched {len(df)} candles.")
    return df

# --- 2. Processing Logic (Rolling & Discretization) ---

def process_candles(df, window=1440):
    """
    Applies the '1440 universes' logic:
    1. Resample to rolling 1D candles (shift open, rolling max/min).
    2. Normalize (Open=1).
    3. Discretize Close into 8 bins relative to High-Low.
    """
    # 1. Rolling 1D representation
    # Open is the open from (window-1) minutes ago
    # High/Low are rolling max/min of window
    daily = pd.DataFrame(index=df.index)
    daily['close'] = df['close']
    daily['open'] = df['open'].shift(window - 1)
    daily['high'] = df['high'].rolling(window=window).max()
    daily['low'] = df['low'].rolling(window=window).min()
    
    # Drop startup NaNs
    daily.dropna(inplace=True)
    
    # 2. Normalize (Center Open at 1)
    shift = daily['open'] - 1
    norm_high = daily['high'] - shift
    norm_low = daily['low'] - shift
    norm_close = daily['close'] - shift
    
    # 3. Discretize
    rng = norm_high - norm_low
    rng = rng.replace(0, 1e-9) # Safety
    section_size = rng / 8
    
    # Binning: floor((Close - Low) / SectionSize)
    bins = ((norm_close - norm_low) / section_size).astype(int)
    bins = bins.clip(0, 7)
    
    daily['state'] = bins
    daily['norm_low'] = norm_low
    daily['section_size'] = section_size
    
    # Prepare features for prediction:
    # We need sequences of states. 
    # S1 was 2 days ago (T-2880), S2 was 1 day ago (T-1440).
    daily['s1'] = daily['state'].shift(2880)
    daily['s2'] = daily['state'].shift(1440)
    daily['prev_close'] = daily['close'].shift(1440)
    
    return daily

# --- 3. Training & Prediction ---

def train_model(train_df):
    """
    Builds the probability map from the training set.
    Map Key: (s1, s2) -> Value: Most frequent next state (target)
    """
    print("Training probability map...")
    # Create triplets (s1, s2) -> target (current state)
    data = train_df[['s1', 's2', 'state']].dropna().copy()
    data.columns = ['s1', 's2', 'target']
    
    # Count frequency of each target for every (s1, s2) pair
    counts = data.groupby(['s1', 's2', 'target']).size().reset_index(name='count')
    
    # Select the target with the highest count for each pair
    best = counts.sort_values('count', ascending=False).drop_duplicates(subset=['s1', 's2'])
    
    # Convert to dict for fast lookup
    prob_map = {}
    for _, row in best.iterrows():
        prob_map[(int(row['s1']), int(row['s2']))] = int(row['target'])
        
    return prob_map

def evaluate_model(test_df, prob_map, label="Test Data"):
    """
    Runs the prediction logic on a dataset using the provided probability map.
    """
    # Filter only valid rows (must have history for s1, s2)
    valid_df = test_df.dropna(subset=['s1', 's2', 'prev_close']).copy()
    
    if len(valid_df) == 0:
        print(f"[{label}] No valid sequences found (insufficient history?).")
        return

    results = []
    
    for idx, row in valid_df.iterrows():
        key = (int(row['s1']), int(row['s2']))
        
        # 1. Look up prediction
        if key not in prob_map:
            continue # Skip unknown sequences
            
        pred_bin = prob_map[key]
        
        # 2. Determine Predicted Direction
        # Open is normalized to 1.
        # Predicted Bin Center (Normalized)
        pred_center = row['norm_low'] + (pred_bin + 0.5) * row['section_size']
        
        # If predicted center > 1 (Open), we predict UP. Else DOWN.
        pred_dir = 1 if pred_center > 1.0 else -1
        
        # 3. Determine Actual Direction
        actual_ret = (row['close'] - row['prev_close']) / row['prev_close']
        actual_dir = 1 if actual_ret > 0 else -1
        
        # 4. Score
        is_correct = (pred_dir == actual_dir)
        pnl = actual_ret if pred_dir == 1 else -actual_ret
        
        results.append({'correct': is_correct, 'pnl': pnl})
        
    res_df = pd.DataFrame(results)
    
    if len(res_df) > 0:
        accuracy = res_df['correct'].mean()
        total_pnl = res_df['pnl'].sum()
        print(f"[{label}] Trades: {len(res_df)} | Accuracy: {accuracy:.2%} | Total PnL: {total_pnl:.4f}")
    else:
        print(f"[{label}] No trades executed.")

# --- 4. Main Execution ---

if __name__ == "__main__":
    # A. Train on Local CSV
    try:
        raw_df = load_local_data('/app/data/ethohlc1m.csv')
        
        # Split 70/30
        split_idx = int(len(raw_df) * 0.70)
        train_raw = raw_df.iloc[:split_idx]
        test_raw = raw_df.iloc[split_idx:]
        
        # Process and Train
        train_processed = process_candles(train_raw)
        prob_map = train_model(train_processed)
        print(f"Map learned {len(prob_map)} patterns.")
        
        # Test on Local Split
        test_processed = process_candles(test_raw)
        evaluate_model(test_processed, prob_map, label="Local CSV Test (30%)")
        
    except FileNotFoundError:
        print("Local CSV not found. Skipping CSV training/testing...")
        # For demonstration if file is missing, we initialize an empty map or handle error
        prob_map = {} 

    # B. Fetch & Test on Live Binance Data
    print("\n--- Starting Live Data Test ---")
    
    # 1. Fetch
    live_df = fetch_binance_data(symbol="ETHUSDT", interval="1m", days=7)
    
    if len(live_df) > 3000: # Need at least ~2 days (2880 mins) for one prediction sequence
        # 2. Process
        live_processed = process_candles(live_df)
        
        # 3. Evaluate (using the map trained on CSV)
        if prob_map:
            evaluate_model(live_processed, prob_map, label="Binance Live Data (Last 7 Days)")
        else:
            print("Model was not trained (CSV missing), cannot test on live data.")
    else:
        print("Insufficient data fetched from Binance to form sequences.")
