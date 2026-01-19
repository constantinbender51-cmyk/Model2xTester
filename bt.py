import pandas as pd
import numpy as np
from collections import defaultdict

def load_and_prep_data(filepath):
    """
    Loads data and creates a rolling 1D representation for every minute
    to simulate the 1440 universes efficiently.
    """
    print("Loading data...")
    # Load data (assuming standard headers, if no headers, names need to be adjusted)
    df = pd.read_csv(filepath)
    
    # Ensure standard column names and datetime index
    df.columns = [c.lower() for c in df.columns]
    # Simple check for standard OHLC names
    rename_map = {}
    for c in df.columns:
        if 'open' in c: rename_map[c] = 'open'
        elif 'high' in c: rename_map[c] = 'high'
        elif 'low' in c: rename_map[c] = 'low'
        elif 'close' in c: rename_map[c] = 'close'
        elif 'date' in c or 'time' in c: rename_map[c] = 'datetime'
    
    df.rename(columns=rename_map, inplace=True)
    
    # Check if datetime exists, otherwise assume index or create generic
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    
    # Sort to ensure chronological order
    df.sort_index(inplace=True)
    
    return df

def create_rolling_candles(df, window=1440):
    """
    Creates 1D candles for every single minute (sliding window).
    This covers all 1440 'universes' simultaneously.
    """
    print("Generating rolling 1D candles (1440 universes)...")
    
    # Construct 1D candles ending at every minute
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
    # 1. Normalize: Center around Open = 1
    # We subtract (Open - 1) from H, L, C.
    # Note: We don't actually need to modify the dataframe columns permanently 
    # if we just calculate the bin, but we need normalized Close for PnL logic later? 
    # Actually PnL is based on raw price returns. We only normalize for the 'State'.
    
    # Shift factor to make Open = 1
    shift_factor = df['open'] - 1
    
    # Normalized values (Open becomes 1)
    norm_high = df['high'] - shift_factor
    norm_low = df['low'] - shift_factor
    norm_close = df['close'] - shift_factor
    # norm_open is always 1, so we don't calculate it
    
    # 2. Discretize
    # Divide High-Low range into 8 sections
    rng = norm_high - norm_low
    
    # Handle cases where High == Low (rare but possible in low liquidity)
    rng = rng.replace(0, 1e-9) 
    
    section_size = rng / 8
    
    # Calculate which bin the Close falls into
    # Bin 0 starts at Low. Bin 7 ends at High.
    # Formula: floor((Close - Low) / SectionSize)
    bins = ((norm_close - norm_low) / section_size).astype(int)
    
    # Clip to 0-7 (Close == High can result in 8 due to float precision)
    bins = bins.clip(0, 7)
    
    df['state'] = bins
    # Store normalized close diff for prediction direction calculation later
    # (Predicted Close State -> translate back to Direction)
    df['norm_close'] = norm_close
    df['norm_low'] = norm_low
    df['section_size'] = section_size
    
    return df

def build_probability_map(train_df):
    """
    Scans every universe (every minute offset) for 3-candle sequences.
    
    Sequence definition:
    At index T (representing a daily candle ending at T),
    Candle 1: Ends at T - 2880 (2 days ago)
    Candle 2: Ends at T - 1440 (1 day ago)
    Candle 3: Ends at T (Today)
    """
    print("Building probability map...")
    
    # We look at triplets spaced by 1440 minutes
    # c1_state = train_df['state'].shift(2880)
    # c2_state = train_df['state'].shift(1440)
    # c3_state = train_df['state']
    
    # Create a DataFrame for the sequences
    seq_df = pd.DataFrame({
        's1': train_df['state'].shift(2880),
        's2': train_df['state'].shift(1440),
        'target': train_df['state']
    }).dropna()
    
    # Count occurrences of (s1, s2) -> target
    counts = seq_df.groupby(['s1', 's2', 'target']).size().reset_index(name='count')
    
    # Find best target for every (s1, s2)
    # Sort by count desc, drop duplicates keeping top
    best_sequences = counts.sort_values('count', ascending=False).drop_duplicates(subset=['s1', 's2'], keep='first')
    
    # Create map: (s1, s2) -> best_target
    prob_map = {}
    for _, row in best_sequences.iterrows():
        prob_map[(int(row['s1']), int(row['s2']))] = int(row['target'])
        
    return prob_map

def run_backtest():
    # 1. Load Data
    raw_df = load_and_prep_data('/app/data/ethohlc1m.csv')
    
    # 2. Split 70/30 (Time-based split on raw data)
    split_idx = int(len(raw_df) * 0.70)
    train_raw = raw_df.iloc[:split_idx]
    test_raw = raw_df.iloc[split_idx:]
    
    print(f"Train size: {len(train_raw)} minutes")
    print(f"Test size: {len(test_raw)} minutes")

    # 3. Process Training Data
    # Create rolling daily candles
    train_candles = create_rolling_candles(train_raw)
    train_candles = discretize_candles(train_candles)
    
    # 4. Build Probability Map
    # This implicitly handles all 1440 universes because 'train_candles' 
    # contains a daily candle ending at every single minute.
    prediction_map = build_probability_map(train_candles)
    print(f"Learned {len(prediction_map)} unique 2-candle sequences.")

    # 5. Process Test Data
    test_candles = create_rolling_candles(test_raw)
    test_candles = discretize_candles(test_candles)
    
    # 6. Evaluate
    # We need the previous two candles (spaced by 1440m) to predict the current one
    # S1 (T-2880), S2 (T-1440) -> Predict S3 (T)
    
    # Align features for vectorization
    test_candles['s1'] = test_candles['state'].shift(2880)
    test_candles['s2'] = test_candles['state'].shift(1440)
    test_candles['prev_close'] = test_candles['close'].shift(1440) # Close of S2
    
    # Drop rows where we don't have enough history
    eval_df = test_candles.dropna(subset=['s1', 's2', 'prev_close']).copy()
    
    total_trades = 0
    correct_direction = 0
    total_pnl = 0.0
    
    # Iterate is necessary here or complex mapping
    # Given the map is a dict, apply is reasonable
    
    def predict_and_score(row):
        key = (int(row['s1']), int(row['s2']))
        
        # If sequence was never seen in training, we skip (or you could assume flat)
        if key not in prediction_map:
            return None, 0
            
        pred_bin = prediction_map[key]
        
        # Determine Predicted Direction
        # We need to reconstruct what price 'pred_bin' implies relative to prev_close
        # We know: bin = (NormClose - NormLow) / SectionSize
        # So NormClose approx = NormLow + (bin + 0.5) * SectionSize
        # Since NormClose = Close - (Open - 1)
        # And Open of current candle = Close of prev candle (row['prev_close'])
        
        # Wait, the user said: "Open of the second candle is close of the first"
        # So Open_current = Prev_Close
        # We normalized current candle such that Open=1.
        # So in normalized terms, Open=1.
        # We need to know if the Predicted Normalized Close is > 1 or < 1.
        
        # Reconstruct approximate normalized close for the predicted bin
        # We use the current candle's NormLow and SectionSize.
        # Note: We are using "cheat" info here (current Low/High) to map the bin back to a price?
        # The user's prompt implies we predict the "Shape/Bin". 
        # A bin of 0 (near Low) vs bin of 7 (near High) implies direction *if we know where High/Low are*.
        # BUT we don't know today's High/Low before it happens.
        
        # Interpretation of User Logic:
        # "look up the continuation and determine directional correctness based on the actual candle following"
        # "directional accuracy is whether close moved in the direction we predicted"
        
        # If we predict a specific bin, do we assume the range (High/Low) is similar to yesterday? 
        # Or is the "state" intrinsically directional?
        # If the bins are 0..7 relative to High/Low:
        # If Open is 1. If we predict a bin that corresponds to a price > 1, it's UP.
        # The relationship between Open (1) and the bins depends on where Open sits in the High-Low range.
        # In our normalization: NormOpen = 1.
        # We check if the predicted bin center is > 1 or < 1.
        
        # Current candle's specific range properties (which we technically predict implicitly or assume known for 'correctness' check context):
        # We can simply check if the *Predicted Bin* represents a value higher than the *Open Bin*.
        # Open is always 1.
        # Where is 1 in the current candle's bin structure?
        # open_bin = (1 - row['norm_low']) / row['section_size']
        
        open_val_norm = 1.0
        # Determine the normalized price range of the predicted bin
        pred_bin_center_norm = row['norm_low'] + (pred_bin + 0.5) * row['section_size']
        
        # Predicted direction
        if pred_bin_center_norm > open_val_norm:
            pred_dir = 1 # UP
        elif pred_bin_center_norm < open_val_norm:
            pred_dir = -1 # DOWN
        else:
            pred_dir = 0
            
        # Actual Direction
        # Close - Open
        actual_change = row['close'] - row['prev_close']
        actual_dir = 1 if actual_change > 0 else (-1 if actual_change < 0 else 0)
        
        # Score
        is_correct = (pred_dir == actual_dir)
        
        # PnL: (Close - PrevClose) / PrevClose
        # User said: "summed over all trades".
        # Assuming we trade in the direction of prediction.
        trade_ret = (row['close'] - row['prev_close']) / row['prev_close']
        
        if pred_dir == 1:
            pnl = trade_ret
        elif pred_dir == -1:
            pnl = -trade_ret # Short
        else:
            pnl = 0
            
        return is_correct, pnl

    # Apply evaluation
    results = eval_df.apply(predict_and_score, axis=1, result_type='expand')
    results.columns = ['correct', 'pnl']
    
    # Filter out skipped (None) trades
    results = results.dropna()
    
    accuracy = results['correct'].mean()
    total_pnl = results['pnl'].sum()
    
    print("-" * 30)
    print(f"Results on Test Data ({len(results)} trades):")
    print(f"Directional Accuracy: {accuracy:.2%}")
    print(f"Total PnL (summed):   {total_pnl:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    run_backtest()
