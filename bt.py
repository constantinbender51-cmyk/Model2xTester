import os
import math
import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from github import Github
from io import BytesIO

# Configuration
DATA_PATH = "/app/data/"
FILE_NAME = "eth_1h_data.csv"
REPO_NAME = "constantinbender51-cmyk/Models"
REMOTE_PLOT_PATH = "plot.png"
A = 0.01

def fetch():
    """
    Fetches 1h ETH/USDT OHLC data from Binance. 
    Loads from local cache if present, otherwise fetches full history.
    """
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    
    file_path = os.path.join(DATA_PATH, FILE_NAME)
    
    if os.path.exists(file_path):
        print(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        # Ensure timestamp is correct type if needed, usually csv loads as int/float or string
        return df

    print("Fetching data from Binance...")
    exchange = ccxt.binance()
    symbol = 'ETH/USDT'
    timeframe = '1h'
    
    # Binance limits fetch per call; pagination required for "all data"
    # Using a simplified since-based fetch
    all_ohlcv = []
    since = exchange.parse8601('2017-08-17T00:00:00Z') # Approximate ETH listing
    now = exchange.milliseconds()
    
    while since < now:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            print(f"Fetched up to {pd.to_datetime(ohlcv[-1][0], unit='ms')}", end='\r')
        except Exception as e:
            print(f"\nError fetching data: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df.to_csv(file_path, index=False)
    print(f"\nData saved to {file_path}")
    return df

def prepare(df):
    """
    Normalizes close price to first close, discretizes by floor(a=0.01),
    splits 80/20, and generates a plot.
    """
    # Keep only close
    prices = df['close'].values
    
    # Normalize
    normalized = prices / prices[0]
    
    # Floor round to nearest A (0.01)
    # Formula: floor(value / step) * step
    discretized = np.floor(normalized / A) * A
    
    # Split 80/20
    split_idx = int(len(discretized) * 0.8)
    train_data = discretized[:split_idx]
    test_data = discretized[split_idx:]
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(discretized, label='Normalized & Discretized Price')
    plt.axvline(x=split_idx, color='r', linestyle='--', label='Train/Test Split')
    plt.title('ETH 1h Normalized/Discretized Data')
    plt.legend()
    
    # Save plot to buffer for upload
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    return train_data, test_data, buf

def upload(image_buffer, repo_name, file_path):
    """
    Uploads the image buffer to the specified GitHub repository.
    """
    load_dotenv()
    token = os.getenv("PAT")
    
    if not token:
        print("Error: PAT not found in .env")
        return

    g = Github(token)
    
    try:
        repo = g.get_repo(repo_name)
        content = image_buffer.getvalue()
        
        try:
            # Check if file exists to update
            contents = repo.get_contents(file_path)
            repo.update_file(contents.path, "Update plot", content, contents.sha)
            print(f"Plot updated at https://github.com/{repo_name}/{file_path}")
        except:
            # Create new file if not exists
            repo.create_file(file_path, "Initial plot upload", content)
            print(f"Plot created at https://github.com/{repo_name}/{file_path}")
            
    except Exception as e:
        print(f"Upload failed: {e}")

def map_sequences(data, input_len=5):
    """
    Maps sequences of length (input_len) to the next value (sequence of 6).
    Returns a dictionary of probabilities.
    Structure: { (v1,v2,v3,v4,v5): { next_val1: prob, next_val2: prob } }
    """
    sequences = {}
    
    # Create windows of 6 (5 input + 1 target)
    for i in range(len(data) - input_len):
        seq = tuple(data[i:i+input_len]) # Key must be hashable
        target = data[i+input_len]
        
        if seq not in sequences:
            sequences[seq] = {}
        
        if target not in sequences[seq]:
            sequences[seq][target] = 0
        sequences[seq][target] += 1
        
    # Convert counts to probabilities
    model = {}
    for seq, targets in sequences.items():
        total = sum(targets.values())
        model[seq] = {k: v / total for k, v in targets.items()}
        
    return model

def pred(input_seq, model):
    """
    Takes input tuple of 5. Returns the 6th value with highest probability.
    Returns None if sequence was not seen in training.
    """
    if input_seq in model:
        # Find key with max value
        options = model[input_seq]
        best_next = max(options, key=options.get)
        return best_next
    return None

def test(test_data, model):
    """
    Tests on test data.
    Calculates PnL and the specific accuracy metric requested.
    Saves a result plot locally (since remote upload was done in prepare, 
    but logic can be reused).
    """
    input_len = 5
    pnl_history = []
    hits = 0
    misses = 0
    ignored = 0 # -a < pnl < a
    
    predictions = []
    actuals = []

    # Iterate through test set
    for i in range(len(test_data) - input_len):
        current_seq = tuple(test_data[i:i+input_len])
        actual_next = test_data[i+input_len]
        current_price = current_seq[-1]
        
        predicted_next = pred(current_seq, model)
        
        if predicted_next is not None:
            # Logic: If model predicts, we take that trade? 
            # Or is PnL based on the accuracy of the prediction?
            # Standard PnL: (Exit - Entry) / Entry. 
            # Here: We assume we bought at current_price and sold at actual_next.
            # But the metric implies we only care if PnL > A. 
            
            # Let's define PnL as the difference between Actual Next and Current 
            # IF the model predicted the correct direction.
            # However, simpler interpretation: PnL of the *predicted move* vs reality.
            
            # Implementation:
            # We assume a Long strategy if Predicted > Current.
            # We assume a Short strategy if Predicted < Current.
            # If Predicted == Current, no trade.
            
            trade_pnl = 0
            if predicted_next > current_price:
                # Long
                trade_pnl = actual_next - current_price 
            elif predicted_next < current_price:
                # Short
                trade_pnl = current_price - actual_next
            
            pnl_history.append(trade_pnl)
            
            # Metric Calculation
            if trade_pnl > A:
                hits += 1
            elif trade_pnl < -A:
                misses += 1
            else:
                ignored += 1
                
            predictions.append(predicted_next)
        else:
            predictions.append(np.nan) # Unknown state
            pnl_history.append(0)
        
        actuals.append(actual_next)

    # Calculate Metric
    # times pnl > a / (times pnl > a + times pnl < -a)
    denominator = hits + misses
    metric = (hits / denominator) if denominator > 0 else 0
    
    print(f"\n--- Test Results ---")
    print(f"Total Trades Simulated: {len(pnl_history)}")
    print(f"PnL > {A}: {hits}")
    print(f"PnL < -{A}: {misses}")
    print(f"Metric (Wins / Wins+Losses): {metric:.4f}")
    
    # Plot PnL curve
    cumulative_pnl = np.cumsum(pnl_history)
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_pnl, label='Cumulative PnL (Test)')
    plt.title(f'Strategy PnL (Metric: {metric:.2f})')
    plt.legend()
    plt.savefig('test_pnl_plot.png')
    print("Test PnL plot saved to test_pnl_plot.png")

def main():
    # 1. Fetch
    df = fetch()
    
    # 2. Prepare
    train, test_data, plot_buf = prepare(df)
    
    # 3. Upload Plot
    upload(plot_buf, REPO_NAME, REMOTE_PLOT_PATH)
    
    # 4. Map Sequences
    print("Mapping sequences (Training)...")
    model = map_sequences(train)
    
    # 5. Example Prediction
    # Take last 5 from training to predict next
    sample_input = tuple(train[-5:])
    prediction = pred(sample_input, model)
    
    print("\n--- Sample Prediction ---")
    print(f"Input Sequence: {sample_input}")
    if prediction is not None:
        # Output 6 starting with 5 with highest probability
        full_sequence = list(sample_input)
        full_sequence.append(prediction)
        print(f"Highest Prob 6-Seq: {full_sequence}")
    else:
        print("Sequence not found in training map.")

    # 6. Test
    test(test_data, model)

if __name__ == "__main__":
    main()
