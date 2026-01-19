import sys
import pickle
import time
import random
import http.server
import socketserver
import numpy as np
import pandas as pd
import ccxt
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from collections import Counter
from huggingface_hub import hf_hub_download

# --- CONFIGURATION ---
ASSETS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT', 
    'AVAX/USDT', 'DOT/USDT', 'LTC/USDT', 'BCH/USDT',
    'LINK/USDT', 'UNI/USDT', 'AAVE/USDT', 'NEAR/USDT', 
    'FIL/USDT', 'ALGO/USDT', 'XLM/USDT', 'EOS/USDT',
    'DOGE/USDT', 'SHIB/USDT', 'SAND/USDT'
]

HF_REPO_ID = "Llama26051996/Models" 
HF_FOLDER = "model2x"
TIMEFRAME = '30m'
PORT = 8000
HTML_FILE = "index.html"
PLOT_FILE = "equity_curve.png"

# --- HELPER FUNCTIONS ---

def get_model_filename(symbol):
    return f"{symbol.split('/')[0].lower()}.pkl"

def optimize_model_patterns(configs):
    """
    SPEEDUP TRICK 1: 
    Pre-calculate the majority vote for every pattern in the model.
    This eliminates the need to run Counter() inside the trading loop.
    """
    print("[*] Optimizing model for fast inference...")
    for cfg in configs:
        fast_patterns = {}
        for seq, outcomes in cfg['patterns'].items():
            if not outcomes: continue
            
            # If it's already an int (some saved models), use it
            if isinstance(outcomes, (int, float, np.number)):
                fast_patterns[seq] = int(outcomes)
                continue
                
            # Otherwise, resolve the list of outcomes to a single vote now
            vote = Counter(outcomes).most_common(1)[0][0]
            fast_patterns[seq] = vote
            
        cfg['patterns'] = fast_patterns # Replace with optimized dict
    return configs

def fetch_history(symbol, since_ts):
    exchange = ccxt.binance()
    all_ohlcv = []
    print(f"[*] Fetching data for {symbol} since {datetime.fromtimestamp(since_ts/1000, tz=timezone.utc)}...")
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since_ts, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            if last_ts == since_ts: break
            since_ts = last_ts + 1
            time.sleep(exchange.rateLimit / 1000)
            if last_ts > (time.time() * 1000) - 60000: break
        except Exception as e:
            print(f"[!] Error fetching data: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# --- FAST BACKTEST ENGINE ---

def run_fast_backtest():
    # 1. Setup
    selected_asset = random.choice(ASSETS)
    print(f"[*] Selected Asset: {selected_asset}")
    
    fname = get_model_filename(selected_asset)
    try:
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=f"{HF_FOLDER}/{fname}", local_dir=".")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
    except Exception as e:
        print(f"[!] Model error: {e}")
        sys.exit(1)

    anchor_price = model_data['initial_price']
    configs = optimize_model_patterns(model_data['ensemble_configs']) # Optimize here
    bucket_size = configs[0]['step_size'] if configs else 0.001

    # 2. Fetch Data
    start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    since_ts = int(start_date.timestamp() * 1000)
    df = fetch_history(selected_asset, since_ts)
    if df.empty: sys.exit(1)

    closes = df['close'].values
    timestamps = df['datetime'].values
    n_candles = len(closes)

    # 3. SPEEDUP TRICK 2: Vectorized Pre-calculation
    # Calculate grids for the ENTIRE history at once
    print("[*] Pre-calculating logic grids...")
    log_prices = np.log(closes / anchor_price)
    
    # Pre-compute integer grids for each config
    # list of numpy arrays, one per config
    config_grids = []
    for cfg in configs:
        # Floor division and cast to int32 for speed/memory
        grid = np.floor(log_prices / cfg['step_size']).astype(np.int32)
        config_grids.append(grid)

    # 4. Fast Loop (Generation of Signals)
    # We only loop to generate the 'Signal' column. PnL is calculated later.
    signals = np.zeros(n_candles, dtype=int) # 1=Long, -1=Short, 0=Neutral
    warmup = 50
    
    print(f"[*] Running fast simulation on {n_candles} candles...")
    start_time = time.time()

    # We iterate through time 'i'
    # Optimizing the inner loop is critical
    for i in range(warmup, n_candles - 1):
        up, down = 0, 0
        
        # Iterate configs
        for idx, cfg in enumerate(configs):
            seq_len = cfg['seq_len']
            
            # Direct slice from pre-computed grid
            # We want the sequence ending at 'i'
            # grid[idx] is the full array for this config
            pattern_seq = tuple(config_grids[idx][i - seq_len + 1 : i + 1])
            
            # O(1) Lookup
            pred_lvl = cfg['patterns'].get(pattern_seq)
            
            if pred_lvl is not None:
                current_lvl = config_grids[idx][i]
                if pred_lvl > current_lvl:
                    up += 1
                elif pred_lvl < current_lvl:
                    down += 1
        
        # Vote Logic
        if up > 0 and down == 0:
            signals[i] = 1 # LONG
        elif down > 0 and up == 0:
            signals[i] = -1 # SHORT
            
    print(f"[*] Simulation Loop Time: {time.time() - start_time:.4f}s")

    # 5. SPEEDUP TRICK 3: Vectorized PnL & Stats
    # Everything below happens instantly using array math
    
    # Shift signals: Signal at 'i' executes on 'i' to 'i+1'
    # We want to align signal[i] with return[i+1]
    
    # Calculate returns for every candle
    # ret[i] = (close[i] - close[i-1]) / close[i-1]
    # We shift closes to get next candle return relative to current
    next_ret_pct = (np.roll(closes, -1) - closes) / closes
    next_log_ret = np.log(np.roll(closes, -1) / closes)
    
    # Fix last element (garbage due to roll)
    next_ret_pct[-1] = 0
    next_log_ret[-1] = 0

    # Calculate PnL Array
    # signal is 1, -1, or 0.
    pnl_array = signals * next_ret_pct
    
    # Filtering Masks
    valid_signals = signals != 0
    # Flat Outcome: |LogReturn| < bucket_size
    # Note: We use the log return of the move that actually happened
    is_flat_outcome = np.abs(next_log_ret) < bucket_size
    
    # Filtered PnL (for accuracy stats only)
    # We don't filter PnL for equity (you pay for flat moves too), 
    # but we filter for the "Directional Accuracy" metric requested.
    
    # --- Metrics Calculation ---
    
    # 1. Directional Accuracy (Excluding Flat Outcomes)
    # Mask: Signal was active AND Outcome was significant
    sig_active_not_flat = valid_signals & (~is_flat_outcome)
    
    # Correct: PnL > 0 (Signal matched direction)
    correct_mask = sig_active_not_flat & (pnl_array > 0)
    
    count_valid = np.count_nonzero(sig_active_not_flat)
    count_correct = np.count_nonzero(correct_mask)
    dir_acc = (count_correct / count_valid * 100) if count_valid > 0 else 0.0

    # 2. 1% Accuracy
    # Wins > 1%
    wins_1pct = np.count_nonzero(pnl_array > 0.01)
    # Losses < -1%
    losses_1pct = np.count_nonzero(pnl_array < -0.01)
    total_1pct = wins_1pct + losses_1pct
    acc_1pct = (wins_1pct / total_1pct * 100) if total_1pct > 0 else 0.0

    # 3. Equity Curve
    # cumulative product of (1 + pnl)
    equity_curve = np.cumprod(1 + pnl_array) * 1000.0
    final_equity = equity_curve[-2] # Last valid point
    total_ret = ((final_equity - 1000) / 1000) * 100
    
    # 4. DataFrame for Reporting
    res_df = pd.DataFrame({
        'date': timestamps,
        'pnl': pnl_array,
        'equity': equity_curve,
        'signal': signals,
        'valid_for_acc': sig_active_not_flat
    })
    # Remove warmup and last garbage row
    res_df = res_df.iloc[warmup:-1]

    # --- Plotting & HTML ---
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(res_df['date'], res_df['equity'], label='Equity ($)', color='blue')
    plt.title(f"Fast Backtest: {selected_asset}\nDir Acc (Filt): {dir_acc:.2f}% | 1% Acc: {acc_1pct:.2f}%")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(PLOT_FILE)

    # Monthly Stats
    res_df['month'] = res_df['date'].dt.to_period('M')
    monthly_stats = []
    
    for name, group in res_df.groupby('month'):
        # Filtered Accuracy for this month
        m_valid = group['valid_for_acc'].sum()
        m_correct = group[group['valid_for_acc'] & (group['pnl'] > 0)].shape[0]
        m_acc = (m_correct / m_valid * 100) if m_valid > 0 else 0.0
        
        # Count trades (non-neutral signals)
        m_trades = np.count_nonzero(group['signal'])
        
        monthly_stats.append({
            'month': str(name),
            'pnl': group['pnl'].sum() * 100,
            'accuracy': m_acc,
            'trades': m_trades
        })

    # Generate HTML
    global_stats = {
        "dir_acc": dir_acc,
        "valid_dir_trades": count_valid,
        "one_pct_acc": acc_1pct,
        "one_pct_count": total_1pct
    }
    generate_html(selected_asset, monthly_stats, 1000.0, final_equity, global_stats)

def generate_html(asset, stats, start_eq, end_eq, global_stats):
    rows = ""
    for s in stats:
        color = "green" if s['pnl'] > 0 else "red"
        rows += f"""
        <tr>
            <td>{s['month']}</td>
            <td>{s['accuracy']:.2f}%</td>
            <td style="color:{color}; font-weight:bold">{s['pnl']:.2f}%</td>
            <td>{s['trades']}</td>
        </tr>
        """
    
    total_ret = ((end_eq - start_eq) / start_eq) * 100
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fast Backtest Report</title>
        <style>
            body {{ font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
            img {{ max-width: 100%; height: auto; margin-top: 20px; border: 1px solid #ddd; }}
            .summary, .metrics {{ background: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .metric-box {{ display: inline-block; width: 45%; vertical-align: top; }}
        </style>
    </head>
    <body>
        <h1>Fast Backtest: {asset}</h1>
        <div class="summary">
            <h3>Performance</h3>
            <p><strong>Total Return:</strong> {total_ret:.2f}%</p>
            <p><strong>Final Equity:</strong> ${end_eq:.2f}</p>
        </div>
        <div class="metrics">
            <div class="metric-box">
                <h4>Filtered Accuracy</h4>
                <p style="font-size: 1.2em; font-weight: bold;">{global_stats['dir_acc']:.2f}%</p>
                <p style="font-size: 0.8em;">Excludes flat moves < {global_stats['valid_dir_trades']} trades</p>
            </div>
            <div class="metric-box">
                <h4>1% Accuracy</h4>
                <p style="font-size: 1.2em; font-weight: bold;">{global_stats['one_pct_acc']:.2f}%</p>
                <p style="font-size: 0.8em;">Sample: {global_stats['one_pct_count']} trades</p>
            </div>
        </div>
        <img src="{PLOT_FILE}" alt="Equity Curve">
        <h2>Monthly Breakdown</h2>
        <table>
            <thead><tr><th>Month</th><th>Acc (Filt)</th><th>Net PnL</th><th>Trades</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </body>
    </html>
    """
    with open(HTML_FILE, "w", encoding="utf-8") as f: f.write(html)
    print(f"[*] HTML Report generated: {HTML_FILE}")

# --- SERVER ---
class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args): pass

if __name__ == "__main__":
    run_fast_backtest()
    print(f"[*] Starting server at http://localhost:{PORT}")
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try: httpd.serve_forever()
        except KeyboardInterrupt: pass
