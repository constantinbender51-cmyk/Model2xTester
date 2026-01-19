import os
import sys
import pickle
import time
import json
import random
import threading
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

def predict_direction(close_prices, anchor_price, configs):
    """
    Replication of the inference logic from the uploaded file.
    """
    prices_arr = np.array(close_prices)
    # Avoid division by zero or log of zero if data is bad
    if anchor_price <= 0 or np.any(prices_arr <= 0):
        return "NEUTRAL"

    log_prices = np.log(prices_arr / anchor_price)
    
    up, down = 0, 0
    
    for cfg in configs:
        grid = np.floor(log_prices / cfg['step_size']).astype(int)
        
        if len(grid) < cfg['seq_len']: continue
        
        seq = tuple(grid[-cfg['seq_len']:])
        if seq in cfg['patterns']:
            pred_lvl = Counter(cfg['patterns'][seq]).most_common(1)[0][0]
            if pred_lvl > seq[-1]: up += 1
            elif pred_lvl < seq[-1]: down += 1
            
    return "LONG" if (up > 0 and down == 0) else "SHORT" if (down > 0 and up == 0) else "NEUTRAL"

def fetch_history(symbol, since_ts):
    """
    Fetches full history from `since_ts` until now using pagination.
    """
    exchange = ccxt.binance()
    all_ohlcv = []
    
    print(f"[*] Fetching data for {symbol} since {datetime.fromtimestamp(since_ts/1000)}...")
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since_ts, limit=1000)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            
            # Update since_ts to the last timestamp + 1ms to get next page
            last_ts = ohlcv[-1][0]
            if last_ts == since_ts: # Break if no progress
                break
            since_ts = last_ts + 1
            
            # Rate limit sleep
            time.sleep(exchange.rateLimit / 1000)
            
            # If we reached close to current time, stop
            if last_ts > (time.time() * 1000) - 60000:
                break
                
        except Exception as e:
            print(f"[!] Error fetching data: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# --- MAIN BACKTEST LOGIC ---

def run_backtest():
    # 1. Select Random Asset
    selected_asset = random.choice(ASSETS)
    print(f"[*] Selected Asset: {selected_asset}")

    # 2. Download Model
    fname = get_model_filename(selected_asset)
    print(f"[*] Downloading model {fname} from HuggingFace...")
    try:
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=f"{HF_FOLDER}/{fname}", local_dir=".")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
    except Exception as e:
        print(f"[!] Failed to download or load model: {e}")
        sys.exit(1)

    anchor_price = model_data['initial_price']
    configs = model_data['ensemble_configs']

    # 3. Fetch Data (From Jan 1, 2025)
    start_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
    since_ts = int(start_date.timestamp() * 1000)
    df = fetch_history(selected_asset, since_ts)

    if df.empty:
        print("[!] No data fetched. Exiting.")
        sys.exit(1)

    print(f"[*] Backtesting on {len(df)} candles...")

    # 4. Simulation Loop
    initial_equity = 1000.0
    equity = initial_equity
    equity_curve = []
    trades = []
    
    # We need a rolling window. The max sequence length in configs determines the warmup.
    # To be safe, we look at the longest seq_len in configs, usually < 20.
    # Let's use 50 as a safe warmup buffer.
    warmup = 50 
    closes = df['close'].values
    timestamps = df['datetime'].values

    for i in range(warmup, len(df) - 1):
        # Input: Historical closes up to index i (inclusive)
        input_series = closes[i-warmup : i+1] 
        
        # Predict
        pred = predict_direction(input_series, anchor_price, configs)
        
        # Trade execution: Enter at close[i], Exit at close[i+1]
        entry_price = closes[i]
        exit_price = closes[i+1]
        ts = timestamps[i+1]
        
        pnl_pct = 0.0
        
        if pred == "LONG":
            pnl_pct = (exit_price - entry_price) / entry_price
        elif pred == "SHORT":
            pnl_pct = (entry_price - exit_price) / entry_price
        
        # Simple compounding
        equity = equity * (1 + pnl_pct)
        equity_curve.append({'date': ts, 'equity': equity})
        
        # Record trade for monthly stats
        if pred != "NEUTRAL":
            trades.append({
                'date': ts,
                'pnl': pnl_pct,
                'win': 1 if pnl_pct > 0 else 0
            })

    # 5. Analysis & Plotting
    eq_df = pd.DataFrame(equity_curve)
    trade_df = pd.DataFrame(trades)
    
    # -- Plotting --
    plt.figure(figsize=(10, 6))
    plt.plot(eq_df['date'], eq_df['equity'], label='Equity ($)', color='blue')
    plt.title(f"Backtest Equity Curve: {selected_asset} (2025-Now)")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(PLOT_FILE)
    print(f"[*] Plot saved to {PLOT_FILE}")

    # -- Monthly Stats --
    monthly_stats = []
    if not trade_df.empty:
        trade_df['month'] = trade_df['date'].dt.to_period('M')
        grouped = trade_df.groupby('month')
        
        for name, group in grouped:
            total_pnl = group['pnl'].sum() * 100
            accuracy = (group['win'].sum() / len(group)) * 100
            monthly_stats.append({
                'month': str(name),
                'pnl': total_pnl,
                'accuracy': accuracy,
                'trades': len(group)
            })

    # 6. Generate HTML
    generate_html(selected_asset, monthly_stats, initial_equity, equity)

# --- HTML GENERATION ---

def generate_html(asset, stats, start_eq, end_eq):
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
        <title>Backtest Report</title>
        <style>
            body {{ font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
            img {{ max-width: 100%; height: auto; margin-top: 20px; border: 1px solid #ddd; }}
            .summary {{ background: #f9f9f9; padding: 15px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Backtest Report: {asset}</h1>
        <div class="summary">
            <p><strong>Period:</strong> 2025-01-01 to Present</p>
            <p><strong>Initial Equity:</strong> ${start_eq:.2f}</p>
            <p><strong>Final Equity:</strong> ${end_eq:.2f}</p>
            <p><strong>Total Return:</strong> {total_ret:.2f}%</p>
        </div>
        
        <img src="{PLOT_FILE}" alt="Equity Curve">
        
        <h2>Monthly Performance</h2>
        <table>
            <thead>
                <tr>
                    <th>Month</th>
                    <th>Accuracy</th>
                    <th>Net PnL</th>
                    <th>Trade Count</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </body>
    </html>
    """
    
    with open(HTML_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[*] HTML Report generated: {HTML_FILE}")

# --- SERVER ---

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence server logs to keep console clean
        pass

def serve():
    print(f"[*] Starting HTTP server at http://localhost:{PORT}")
    print("[*] Press Ctrl+C to stop.")
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()

if __name__ == "__main__":
    run_backtest()
    serve()
