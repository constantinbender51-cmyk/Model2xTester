import pandas as pd
import numpy as np
import yfinance as yf

def fetch_data():
    """
    Fetches the most recent 7 days of 1m data for ETH-USD from Yahoo Finance.
    """
    print("Fetching most recent week of ETH-USD 1m data...")
    # Fetch 7 days (maximum typically allowed for 1m interval without premium API)
    df = yf.download("ETH-USD", interval="1m", period="7d", progress=False)
    
    # Flatten multi-index columns if they exist (yfinance update quirk)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # Standardize columns
    df.columns = [c.lower() for c in df.columns]
    
    # Ensure we have a datetime index (yfinance usually returns this by default)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    # Drop rows with missing data
    df.dropna(inplace=True)
    
    return df

def create_rolling_candles(df, window=1440):
    """
    Creates 1D candles for every single minute (sliding window).
    """
    # Open is the open 1439 minutes ago
    # High is max of last 1440
    # Low is min of last 1440
    # Close is current close
    
    daily_candles = pd.DataFrame(index=df.index)
    daily_candles['close'] = df['close']
    daily_candles['open'] = df['open'].shift(window - 1)
    daily_candles['high'] = df['high'].rolling(window=window).max()
    daily_candles['low'] = df['low'].rolling(window=window).min()
    
    # Drop NaNs created by the rolling window and shift
    daily_candles.dropna(inplace=True)
    
    return daily_candles

def discretize_candles(df):
    """
    Normalizes (Open=1) and Discretizes Close into 8 bins relative to High-Low.
    """
    # Shift factor to make Open = 1
    shift_factor = df['open'] - 1
    
    # Normalized values (Open becomes 1)
    norm_high = df['high'] - shift_factor
    norm_low = df['low'] - shift_factor
    norm_close = df['close'] - shift_factor
    
    # Discretize High-Low range into 8 sections
    rng = norm_high - norm_low
    rng = rng.replace(0, 1e-9) 
    section_size = rng / 8
    
    # Calculate bin: floor((Close - Low) / SectionSize)
    bins = ((norm_close - norm_low) / section_size).astype(int)
    bins = bins.clip(0, 7)
    
    df['state'] = bins
    df['norm_low'] = norm_low
    df['section_size'] = section_size
    
    return df

def build_probability_map(train_df):
    """
    Scans for 3-candle sequences: S1(T-2880), S2(T-1440), Target(T)
    """
    # Create a DataFrame for the sequences
    seq_df = pd.DataFrame({
        's1': train_df['state'].shift(2880),
        's2': train_df['state'].shift(1440),
        'target': train_df['state']
    }).dropna()
    
    # Count occurrences of (s1, s2) -> target
    counts = seq_df.groupby(['s1', 's2', 'target']).size().reset_index(name='count')
    
    # Keep only highest probability target for every (s1, s2)
    best_sequences = counts.sort_values('count', ascending=False).drop_duplicates(subset=['s1', 's2'], keep='first')
    
    # Map (s1, s2) -> best_target
    prob_map = {}
    for _, row in best_sequences.iterrows():
        prob_map[(int(row['s1']), int(row['s2']))] = int(row['target'])
        
    return prob_map

def predict_and_score_row(row, prediction_map):
    """
    Helper to calculate direction and pnl for a single row.
    """
    key = (int(row['s1']), int(row['s2']))
    
    # Skip unknown sequences
    if key not in prediction_map:
        return np.nan, 0.0
        
    pred_bin = prediction_map[key]
    
    # Determine Predicted Direction relative to Open (which is 1)
    # Reconstruct center of predicted bin in normalized terms
    pred_bin_center_norm = row['norm_low'] + (pred_bin + 0.5) * row['section_size']
    
    # Open is always 1 in normalized space
    if pred_bin_center_norm > 1.0:
        pred_dir = 1  # UP
    elif pred_bin_center_norm < 1.0:
        pred_dir = -1 # DOWN
    else:
        pred_dir = 0
        
    # Actual Direction
    actual_change = row['close'] - row['prev_close']
    actual_dir = 1 if actual_change > 0 else (-1 if actual_change < 0 else 0)
    
    is_correct = (pred_dir == actual_dir)
    
    # PnL: (Close - PrevClose) / PrevClose
    trade_ret = (row['close'] - row['prev_close']) / row['prev_close']
    
    if pred_dir == 1:
        pnl = trade_ret
    elif pred_dir == -1:
        pnl = -trade_ret
    else:
        pnl = 0.0
        
    return 1.0 if is_correct else 0.0, pnl

def run_backtest():
    # 1. Fetch Data
    raw_df = fetch_data()
    
    if len(raw_df) < 5000:
        print("Warning: Not enough data fetched to form valid 3-day sequences.")
        return

    # 2. Split 70/30
    split_idx = int(len(raw_df) * 0.70)
    train_raw = raw_df.iloc[:split_idx]
    test_raw = raw_df.iloc[split_idx:]
    
    print(f"Train samples: {len(train_raw)} | Test samples: {len(test_raw)}")

    # 3. Train
    train_candles = create_rolling_candles(train_raw)
    train_candles = discretize_candles(train_candles)
    prediction_map = build_probability_map(train_candles)
    print(f"Learned {len(prediction_map)} unique 2-candle sequences.")

    # 4. Test
    test_candles = create_rolling_candles(test_raw)
    test_candles = discretize_candles(test_candles)
    
    # Align features (S1 = T-2880, S2 = T-1440)
    test_candles['s1'] = test_candles['state'].shift(2880)
    test_candles['s2'] = test_candles['state'].shift(1440)
    test_candles['prev_close'] = test_candles['close'].shift(1440)
    
    eval_df = test_candles.dropna(subset=['s1', 's2', 'prev_close']).copy()
    
    if eval_df.empty:
        print("No test data left after shifting (need 2 days of history).")
        return

    # 5. Execute Predictions
    # Using list comprehension for speed over .apply
    results = [predict_and_score_row(row, prediction_map) for _, row in eval_df.iterrows()]
    
    # Convert to DataFrame
    res_df = pd.DataFrame(results, columns=['correct', 'pnl'])
    
    # Drop skipped trades (NaNs)
    res_df = res_df.dropna()
    
    if res_df.empty:
        print("No matching sequences found in test data.")
        return

    accuracy = res_df['correct'].mean()
    total_pnl = res_df['pnl'].sum()
    
    print("-" * 30)
    print(f"Results on Test Data ({len(res_df)} trades):")
    print(f"Directional Accuracy: {accuracy:.2%}")
    print(f"Total PnL (summed):   {total_pnl:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    run_backtest()
