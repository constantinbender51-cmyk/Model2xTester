import os
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

# A = 0.01. We scale by 100 to work with natural numbers.
# Threshold A becomes 1 in the scaled domain.
SCALE_FACTOR = 100 
A_SCALED = 1 

def fetch():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    
    file_path = os.path.join(DATA_PATH, FILE_NAME)
    
    if os.path.exists(file_path):
        print(f"Loading data from {file_path}")
        return pd.read_csv(file_path)

    print("Fetching data from Binance...")
    exchange = ccxt.binance()
    all_ohlcv = []
    since = exchange.parse8601('2017-08-17T00:00:00Z') 
    now = exchange.milliseconds()
    
    while since < now:
        try:
            ohlcv = exchange.fetch_ohlcv('ETH/USDT', '1h', since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            print(f"Fetched up to {pd.to_datetime(ohlcv[-1][0], unit='ms')}", end='\r')
        except Exception:
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df.to_csv(file_path, index=False)
    return df

def prepare(df):
    """
    1. Normalize to first close.
    2. Multiply by 100 (1/0.01).
    3. Floor to integer.
    """
    prices = df['close'].values
    normalized = prices / prices[0]
    
    # Scale and Floor: 1.05 -> 105.0 -> 105
    # This represents 'rounding to a floor where a is 0.01' but in integer space
    natural_numbers = np.floor(normalized * SCALE_FACTOR).astype(int)
    
    # Split
    split_idx = int(len(natural_numbers) * 0.8)
    train_data = natural_numbers[:split_idx]
    test_data = natural_numbers[split_idx:]
    
    # Plotting (Display as float for readability)
    plt.figure(figsize=(12, 6))
    plt.plot(natural_numbers / SCALE_FACTOR, label='Normalized & Floored Price')
    plt.axvline(x=split_idx, color='r', linestyle='--', label='Train/Test Split')
    plt.title('ETH 1h Data (Step 0.01)')
    plt.legend()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    return train_data, test_data, buf

def upload(image_buffer, repo_name, file_path):
    load_dotenv()
    token = os.getenv("PAT")
    if not token: return

    g = Github(token)
    try:
        repo = g.get_repo(repo_name)
        content = image_buffer.getvalue()
        try:
            contents = repo.get_contents(file_path)
            repo.update_file(contents.path, "Update plot", content, contents.sha)
        except:
            repo.create_file(file_path, "Initial plot upload", content)
    except Exception as e:
        print(f"Upload error: {e}")

def map_sequences(data, input_len=5):
    sequences = {}
    
    # Data is already pure integers, no float artifacts
    for i in range(len(data) - input_len):
        seq = tuple(data[i:i+input_len])
        target = data[i+input_len]
        
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
    if input_seq in model:
        return max(model[input_seq], key=model[input_seq].get)
    return None

def test(test_data, model):
    input_len = 5
    pnl_history = []
    hits = 0
    misses = 0
    
    for i in range(len(test_data) - input_len):
        current_seq = tuple(test_data[i:i+input_len])
        actual_next = test_data[i+input_len]
        current_price = current_seq[-1]
        
        predicted_next = pred(current_seq, model)
        
        if predicted_next is not None:
            # PnL logic in integers
            trade_pnl = 0
            if predicted_next > current_price:
                trade_pnl = actual_next - current_price 
            elif predicted_next < current_price:
                trade_pnl = current_price - actual_next
            
            pnl_history.append(trade_pnl)
            
            # Use Scaled A (1)
            if trade_pnl > A_SCALED:
                hits += 1
            elif trade_pnl < -A_SCALED:
                misses += 1
        else:
            pnl_history.append(0)

    denominator = hits + misses
    metric = (hits / denominator) if denominator > 0 else 0
    
    print(f"\n--- Test Results (Scaled Units) ---")
    print(f"Trades: {len(pnl_history)}")
    print(f"Wins (> {A_SCALED} unit): {hits}")
    print(f"Losses (< -{A_SCALED} unit): {misses}")
    print(f"Accuracy: {metric:.4f}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(np.cumsum(pnl_history), label='Cumulative PnL (Scaled Units)')
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
    sample_input = tuple(train[-5:])
    prediction = pred(sample_input, model)
    
    print("\n--- Sample Prediction ---")
    print(f"Input (Int): {list(sample_input)}")
    if prediction is not None:
        full_seq = list(sample_input)
        full_seq.append(prediction)
        print(f"Output (Int): {full_seq}")
        # Convert back to float for user reference
        print(f"Output (Float approx): {[x/SCALE_FACTOR for x in full_seq]}")
    else:
        print("Sequence not found.")

    test(test_data, model)

if __name__ == "__main__":
    main()
