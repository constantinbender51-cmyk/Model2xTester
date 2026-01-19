import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import requests
import base64
import io
from dotenv import load_dotenv
from collections import defaultdict, Counter
import time

# Load environment variables
load_dotenv()

class ETHModel:
    def __init__(self):
        self.df = None
        self.train_data = []
        self.test_data = []
        self.sequence_map = defaultdict(Counter)
        self.best_a = 0.01
        self.github_owner = "constantinbender51-cmyk"
        self.github_repo = "Models"
        # path for local caching
        self.data_path = "/app/data/eth_ohlc.csv"

    def fetch(self):
        """Fetch ohlc eth 1h binance all data, save/load from local path"""
        
        # 1. Try to load from local file first
        if os.path.exists(self.data_path):
            print(f"Loading data from {self.data_path}...")
            try:
                self.df = pd.read_csv(self.data_path)
                self.df.sort_values('timestamp', inplace=True)
                print(f"Loaded {len(self.df)} rows from cache.")
                return
            except Exception as e:
                print(f"Error loading cache: {e}. Re-fetching...")

        # 2. Fetch from Binance if no cache or error
        print("Fetching ETH/USDT 1h data from Binance. This may take a minute...")
        exchange = ccxt.binance()
        symbol = 'ETH/USDT'
        timeframe = '1h'
        
        since = exchange.parse8601('2017-08-17T00:00:00Z') 
        all_ohlcv = []
        
        while True:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
                if len(ohlcv) == 0:
                    break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 60 * 60 * 1000 
                
                if since > exchange.milliseconds():
                    break
                    
                time.sleep(exchange.rateLimit / 1000)
                print(f"Fetched {len(all_ohlcv)} candles...", end='\r')
            except Exception as e:
                print(f"\nError fetching data: {e}")
                break
                
        self.df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.df.sort_values('timestamp', inplace=True)
        print(f"\nTotal data fetched: {len(self.df)} rows.")
        
        # 3. Save to local path
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            self.df.to_csv(self.data_path, index=False)
            print(f"Saved data to {self.data_path}")
        except Exception as e:
            print(f"Could not save data to {self.data_path}: {e}")

    def prepare(self, a=None):
        if self.df is None or len(self.df) == 0:
            print("No data to prepare.")
            return [], []

        if a is None:
            a = self.best_a
            
        prices = self.df['close'].values.astype(float)
        start_price = prices[0] if prices[0] != 0 else 1
        normalized = prices / start_price
        
        # Round to floor where a is step
        discretized = np.floor((normalized + 1e-9) / a) * a
        
        # Split 80/20
        split_idx = int(len(discretized) * 0.8)
        self.train_data = discretized[:split_idx]
        self.test_data = discretized[split_idx:]
        
        return self.train_data, self.test_data

    def upload(self, plt_obj, filename):
        """Save plot to GitHub via API (supports folder paths like 'plot/plot.png')"""
        token = os.getenv('PAT')
        if not token:
            print("Error: PAT not found in .env")
            return

        # Save plot to buffer
        buf = io.BytesIO()
        plt_obj.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        content_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        url = f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}/contents/{filename}"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        # Check if file exists to get SHA
        try:
            get_resp = requests.get(url, headers=headers)
            data = {
                "message": f"Update {filename}",
                "content": content_base64
            }
            
            if get_resp.status_code == 200:
                data["sha"] = get_resp.json()['sha']
            
            print(f"Uploading to {filename}...")
            resp = requests.put(url, json=data, headers=headers)
            if resp.status_code in [200, 201]:
                print(f"Successfully uploaded {filename}.")
            else:
                print(f"Failed to upload {filename}: {resp.status_code} {resp.text}")
        except Exception as e:
            print(f"Connection error during upload: {e}")

    def map_sequences(self, data):
        self.sequence_map.clear()
        if len(data) < 6: return

        for i in range(len(data) - 5):
            seq = tuple(data[i : i+5])
            next_val = data[i+5]
            self.sequence_map[seq][next_val] += 1
            
    def pred(self, input_seq):
        seq_key = tuple(input_seq)
        if seq_key in self.sequence_map and self.sequence_map[seq_key]:
            prediction = self.sequence_map[seq_key].most_common(1)[0][0]
            return prediction
        else:
            return input_seq[-1]

    def grid(self):
        print("\nStarting Grid Search for 'a'...")
        a_values = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
        best_score = -float('inf')
        
        for a in a_values:
            train_qs, _ = self.prepare(a)
            self.map_sequences(train_qs)
            
            pnl_greater_a = 0
            pnl_less_neg_a = 0
            
            if len(train_qs) < 6: continue
            
            for i in range(len(train_qs) - 5):
                current_seq = train_qs[i : i+5]
                current_price = train_qs[i+4]
                actual_next = train_qs[i+5]
                
                predicted_next = self.pred(current_seq)
                
                direction = 0
                if predicted_next > current_price + 1e-9:
                    direction = 1
                elif predicted_next < current_price - 1e-9:
                    direction = -1
                
                trade_pnl = (actual_next - current_price) * direction
                
                if trade_pnl > a:
                    pnl_greater_a += 1
                elif trade_pnl < -a:
                    pnl_less_neg_a += 1
            
            score = pnl_greater_a - pnl_less_neg_a
            print(f"a={a:.3f}: Score={score} (+{pnl_greater_a} / -{pnl_less_neg_a})")
            
            if score > best_score:
                best_score = score
                self.best_a = a
        
        print(f"Optimization complete. Best a: {self.best_a}")

    def test(self):
        print(f"\nRunning Test with best a={self.best_a}...")
        train_final, test_final = self.prepare(self.best_a)
        self.map_sequences(train_final)
        
        if len(test_final) < 6:
            print("Not enough test data.")
            return

        pnls = []
        count_pnl_gt_a = 0
        count_pnl_lt_neg_a = 0
        cumulative_pnl = [0]
        
        for i in range(len(test_final) - 5):
            current_seq = test_final[i : i+5]
            current_price = test_final[i+4]
            actual_next = test_final[i+5]
            
            predicted_next = self.pred(current_seq)
            
            direction = 0
            if predicted_next > current_price + 1e-9:
                direction = 1
            elif predicted_next < current_price - 1e-9:
                direction = -1
                
            trade_pnl = (actual_next - current_price) * direction
            pnls.append(trade_pnl)
            cumulative_pnl.append(cumulative_pnl[-1] + trade_pnl)
            
            if trade_pnl > self.best_a:
                count_pnl_gt_a += 1
            elif trade_pnl < -self.best_a:
                count_pnl_lt_neg_a += 1
        
        denominator = count_pnl_gt_a + count_pnl_lt_neg_a
        accuracy = 0
        if denominator > 0:
            accuracy = count_pnl_gt_a / denominator
            
        print(f"Results on Test Data: Accuracy={accuracy:.2%}, Wins={count_pnl_gt_a}, Losses={count_pnl_lt_neg_a}")
        
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_pnl, label='Cumulative PnL', color='green')
        plt.title(f'Strategy PnL (a={self.best_a}) | Acc: {accuracy:.2%}')
        plt.xlabel('Trades')
        plt.ylabel('PnL')
        plt.legend()
        plt.grid(True)
        
        # Save PnL plot to plot/pnl.png
        self.upload(plt, "plot/pnl.png")
        plt.close()

    def main(self):
        self.fetch()
        if self.df is None or len(self.df) == 0: return

        # Plot ALL Data
        print("\nGenerating plot of all data...")
        viz_a = 0.01 
        train_viz, test_viz = self.prepare(viz_a)
        all_data = np.concatenate((train_viz, test_viz))
        split_index = len(train_viz)
        
        plt.figure(figsize=(15, 7))
        plt.plot(all_data, label=f'Normalized Price (a={viz_a})', linewidth=0.5)
        plt.axvline(x=split_index, color='r', linestyle='--', linewidth=1.5, label='Split')
        plt.title(f"All ETH/USDT 1h Data (N={len(all_data)})")
        plt.legend()
        plt.grid(True)
        
        # Save data plot to plot/plot.png
        self.upload(plt, "plot/plot.png")
        plt.close()
        
        self.grid()
        self.test()
        print("\nExit.")

if __name__ == "__main__":
    model = ETHModel()
    model.main()
