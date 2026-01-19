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
    Loads from local cache if present.
    """
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    
    file_path = os.path.join(DATA_PATH, FILE_NAME)
    
    if os.path.exists(file_path):
        print(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        return df

    print("Fetching data from Binance...")
    exchange = ccxt.binance()
    symbol = 'ETH/USDT'
    timeframe = '1h'
    
    all_ohlcv = []
    since = exchange.parse8601('2017-08-17T00:00:00Z') 
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
    return df

def prepare(df):
    """
    Normalizes close price, discretizes by floor(a), and explicitly rounds 
    to remove floating point artifacts.
    """
    prices = df['close'].values
    
    # Normalize
    normalized = prices / prices[0]
    
    # Calculate required precision based on A (e.g., 0.01 -> 2 decimals)
    precision = int(abs(math.log10(A)))
    
    # Floor round to nearest A, then strictly round to precision to fix float artifacts
    # Formula: round(floor(value / step) * step, precision)
    discretized = np.round(np.floor(normalized / A) * A, precision)
    
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
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    return train_data, test_data, buf

def upload(image_buffer, repo_name, file_path):
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
            contents = repo.get_contents(file_path)
            repo.update_file(contents.path, "Update plot", content, contents.sha)
            print(f"Plot updated at https://github.com/{repo_name}/{file_path}")
        except:
            repo.create_file(file_path, "Initial plot upload", content)
            print(f"Plot created at https://github.com/{repo_name}/{file_path}")
    except Exception as e:
        print(f"Upload failed: {e}")

def map_sequences(data, input_len=5):
    """
    Maps sequences of length 5 to the next value.
    Ensures keys are standard Python floats to avoid numpy type representation issues.
    """
    sequences = {}
    
    for i in range(len(data) - input_len):
        # Convert numpy types to standard python floats for clean tuples
        seq = tuple(float(x) for x in data[i:i+input_len])
        target = float(data[i+input_len])
        
        if seq not in sequences:
            sequences[seq] = {}
        
        if target not in sequences[seq]:
            sequences[seq][target] = 0
        sequences[seq][target] += 1
        
    model = {}
    for seq, targets in sequences.items():
        total = sum(targets.values())
        model[seq] = {k: v / total for k, v in targets.items()}
        
    return model

def pred(input_seq, model):
    # Ensure input is standard tuple of floats
    clean_input = tuple(float(x) for x in input_seq)
    
    if clean_input in model:
        options = model[clean_input]
        best_next = max(options, key=options.get)
        return best_next
    return None

def test(test_data, model):
    input_len = 5
    pnl_history = []
    hits = 0
    misses = 0
    
    for i in range(len(test_data) - input_len):
        current_seq = test_data[i:i+input_len]
        actual_next = float(test_data[i+input_len])
        current_price = float(current_seq[-1])
        
        predicted_next = pred(current_seq, model)
        
        if predicted_next is not None:
            trade_pnl = 0
            if predicted_next > current_price:
                trade_pnl = actual_next - current_price 
            elif predicted_next < current_price:
                trade_pnl = current_price - actual_next
            
            pnl_history.append(trade_pnl)
            
            if trade_pnl > A:
                hits += 1
            elif trade_pnl < -A:
                misses += 1
        else:
            pnl_history.append(0)

    denominator = hits + misses
    metric = (hits / denominator) if denominator > 0 else 0
    
    print(f"\n--- Test Results ---")
    print(f"Trades Evaluated: {len(pnl_history)}")
    print(f"Wins (> {A}): {hits}")
    print(f"Losses (< -{A}): {misses}")
    print(f"Accuracy Metric: {metric:.4f}")
    
    cumulative_pnl = np.cumsum(pnl_history)
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_pnl, label='Cumulative PnL')
    plt.title(f'Strategy PnL (Accuracy: {metric:.2f})')
    plt.legend()
    plt.savefig('test_pnl_plot.png')
    print("Test plot saved locally.")

def main():
    df = fetch()
    train, test_data, plot_buf = prepare(df)
    upload(plot_buf, REPO_NAME, REMOTE_PLOT_PATH)
    
    print("Mapping sequences...")
    model = map_sequences(train)
    
    # Sample Prediction
    sample_input = train[-5:]
    prediction = pred(sample_input, model)
    
    print("\n--- Sample Prediction ---")
    # Clean output for display
    print(f"Input: {[float(x) for x in sample_input]}")
    if prediction is not None:
        full_seq = list(sample_input)
        full_seq.append(prediction)
        print(f"Projected 6-Seq: {[float(x) for x in full_seq]}")
    else:
        print("Sequence not found.")

    test(test_data, model)

if __name__ == "__main__":
    main()
