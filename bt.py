import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta

# --- 1. Data Fetching (Live Binance API) ---
def fetch_binance_data(symbol="ETHUSDT", interval="1m", days=7):
    print(f"Fetching last {days} days of {interval} data for {symbol}...")
    
    base_url = "https://api.binance.com/api/v3/klines"
    limit = 1000 # Binance max limit
    
    # Calculate timestamps (ms)
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)
    
    all_data = []
    current_start = start_time
    
    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "limit": limit
        }
        
        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if not data or isinstance(data, dict) and 'code' in data:
                print("Error or end of data reached.")
                break
                
            all_data.extend(data)
            
            # Update start time to the last candle's close time + 1ms
            # data[6] is Close Time, but usually we just step forward by limit * interval
            # Safer to take the last candle's open time + 1m
            last_open_time = data[-1][0]
            current_start = last_open_time + 60000 
            
            # Respect rate limits slightly
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Network error: {e}")
            break
            
    # Create DataFrame
    # Columns: Open Time, Open, High, Low, Close, Volume, ...
    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    
    # Type conversion
    numeric_cols = ['open', 'high', 'low', 'close']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
    df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    
    # Drop duplicates if any overlap
    df = df[~df.index.duplicated(keep='first')]
    
    print(f"Fetched {len(df)} 1m candles.")
    return df[['open', 'high', 'low', 'close']]

# --- 2. Core Logic (Universes & Backtest) ---

def create_rolling_candles(df, window=1440):
    """
    Creates 1D candles for every single minute (sliding window).
    Open(T) = Close(T-1440) -> effectively Open(T) = Open(T-1439) in standard terms?
    User spec: "Open of the second candle is close of the first". 
    For a daily candle ending at T, 'Open' is the price 1440m ago.
    """
    daily_candles = pd.DataFrame(index=df.index)
    daily_candles['close'] = df['close']
    daily_candles['open'] = df['close'].shift(window) # Open = Close of previous day
    # Note: Using shift(window) aligns 'open' with the candle ending at T.
    
    daily_candles['high'] = df['high'].rolling(window=window).max()
    daily_candles['low'] = df['low'].rolling(window=window).min()
    
    daily_candles.dropna(inplace=True)
    return daily_candles

def discretize_candles(df):
    """
    Normalizes (Open=1) and Discretizes Close into 8 bins relative to High-Low.
    """
    # Shift factor to make Open = 1
    shift_factor = df['open'] - 1
    
    norm_high = df['high'] - shift_factor
    norm_low = df['low'] - shift_factor
    norm_close = df['close'] - shift_factor
    
    # Range
    rng = norm_high - norm_low
    rng = rng.replace(0, 1e-9)
    
    section_size = rng / 8
    
    # Discretize Close
    bins = ((norm_close - norm_low) / section_size).astype(int)
    bins = bins.clip(0, 7)
    
    df['state'] = bins
    df['norm_low'] = norm_low
    df['section_size'] = section_size
    
    return df

def build_probability_map(train_df):
    """
    Map: (S_t-2880, S_t-1440) -> Most Probable S_t
    """
    seq_df = pd.DataFrame({
        's1': train_df['state'].shift(2880),
        's2': train_df['state'].shift(1440),
        'target': train_df['state']
    }).dropna()
    
    # Count occurrences
    counts = seq_df.groupby(['s1', 's2', 'target']).size().reset_index(name='count')
    
    # Select highest probability target
    best_sequences = counts.sort_values('count', ascending=False).drop_duplicates(subset=['s1', 's2'], keep='first')
    
    prob_map = {}
    for _, row in best_sequences.iterrows():
        prob_map[(int(row['s1']), int(row['s2']))] = int(row['target'])
        
    return prob_map

def run_test():
    # 1. Fetch Data
    raw_df = fetch_binance_data(days=7) # 1 week
    
    if len(raw_df) < 3000:
        print("Insufficient data for 1440m windows.")
        return

    # 2. Split 70/30
    split_idx = int(len(raw_df) * 0.70)
    train_raw = raw_df.iloc[:split_idx]
    test_raw = raw_df.iloc[split_idx:]
    
    print(f"Train raw: {len(train_raw)} | Test raw: {len(test_raw)}")

    # 3. Prepare Train
    train_candles = create_rolling_candles(train_raw)
    train_candles = discretize_candles(train_candles)
    
    # 4. Train
    prediction_map = build_probability_map(train_candles)
    print(f"Trained: {len(prediction_map)} sequences mapped.")

    # 5. Prepare Test
    test_candles = create_rolling_candles(test_raw)
    test_candles = discretize_candles(test_candles)
    
    # Align features (S1, S2, PrevClose)
    test_candles['s1'] = test_candles['state'].shift(2880)
    test_candles['s2'] = test_candles['state'].shift(1440)
    test_candles['prev_close'] = test_candles['close'].shift(1440) 
    
    eval_df = test_candles.dropna(subset=['s1', 's2', 'prev_close']).copy()
    
    if len(eval_df) == 0:
        print("Not enough test data to form sequences (need 2 days history in test set or carry over).")
        print("Note: In a pure split, the first 2 days of 'Test' are burned waiting for history.")
        return

    # 6. Predict & Score
    def get_score(row):
        key = (int(row['s1']), int(row['s2']))
        
        if key not in prediction_map:
            return None, 0.0
            
        pred_bin = prediction_map[key]
        
        # Calculate Predicted Direction
        # Normalized Open is always 1.
        # Find center of predicted bin in normalized prices
        pred_norm_price = row['norm_low'] + (pred_bin + 0.5) * row['section_size']
        
        # Direction: Predicted Price vs Open (1.0)
        pred_dir = 1 if pred_norm_price > 1.0 else (-1 if pred_norm_price < 1.0 else 0)
        
        # Actual Direction
        actual_ret = (row['close'] - row['prev_close']) / row['prev_close']
        actual_dir = 1 if actual_ret > 0 else (-1 if actual_ret < 0 else 0)
        
        is_correct = (pred_dir == actual_dir)
        
        # PnL logic: Trade in predicted direction
        trade_pnl = actual_ret if pred_dir == 1 else (-actual_ret if pred_dir == -1 else 0)
        
        return is_correct, trade_pnl

    results = eval_df.apply(get_score, axis=1, result_type='expand')
    results.columns = ['correct', 'pnl']
    results = results.dropna()
    
    acc = results['correct'].mean()
    pnl = results['pnl'].sum()
    
    print("="*40)
    print(f"Backtest Results (Last Week Data)")
    print(f"Trades Executed: {len(results)}")
    print(f"Accuracy:        {acc:.2%}")
    print(f"Total PnL:       {pnl:.4f}")
    print("="*40)

if __name__ == "__main__":
    run_test()
