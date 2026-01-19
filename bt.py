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
        self.train_tokens = []
        self.train_returns = []
        self.test_tokens = []
        self.test_returns = []
        self.sequence_map = defaultdict(Counter)
        self.best_a = 0.01
        self.github_owner = "constantinbender51-cmyk"
        self.github_repo = "Models"
        self.data_path = "/app/data/eth_ohlc.csv"

    def fetch(self):
        """Fetch ohlc eth 1h binance all data, save/load from local path"""
        if os.path.exists(self.data_path):
            print(f"Loading data from {self.data_path}...")
            try:
                self.df = pd.read_csv(self.data_path)
                self.df.sort_values('timestamp', inplace=True)
                print(f"Loaded {len(self.df)} rows from cache.")
                return
            except Exception as e:
                print(f"Error loading cache: {e}. Re-fetching...")

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
        
        try:
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            self.df.to_csv(self.data_path, index=False)
            print(f"Saved data to {self.data_path}")
        except Exception as e:
            print(f"Could not save data to {self.data_path}: {e}")

    def prepare(self, a=None):
        if self.df is None or len(self.df) == 0:
            print("No data to prepare.")
            return [], [], [], []

        if a is None:
            a = self.best_a
            
        prices = self.df['close'].values.astype(float)
        
        # 1. Calculate Returns (Stationary Data)
        # diff / prices[:-1] gives us the % change for each candle
        actual_returns = np.diff(prices) / (prices[:-1] + 1e-9)
        
        # 2. Ternary Discretization (Limit lowest bin to 'a')
        # 1 = Bullish (> a)
        # -1 = Bearish (< -a)
        # 0 = Flat
        tokens = np.zeros_like(actual_returns)
        tokens[actual_returns > a] = 1
        tokens[actual_returns < -a] = -1
        
        # Split 50/50
        split_idx = int(len(tokens) * 0.5)
        
        self.train_tokens = tokens[:split_idx]
        self.train_returns = actual_returns[:split_idx]
        
        self.test_tokens = tokens[split_idx:]
        self.test_returns = actual_returns[split_idx:]
        
        return self.train_tokens, self.train_returns, self.test_tokens, self.test_returns

    def upload(self, plt_obj, filename):
        """Save plot to GitHub via API"""
        token = os.getenv('PAT')
        if not token:
            print("Error: PAT not found in .env")
            return

        buf = io.BytesIO()
        plt_obj.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        content_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        url = f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}/contents/{filename}"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
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

    def map_sequences(self, tokens):
        """Build map from discrete tokens (-1, 0, 1)"""
        self.sequence_map.clear()
        if len(tokens) < 6: return

        for i in range(len(tokens) - 5):
            seq = tuple(tokens[i : i+5])
            next_val = tokens[i+5]
            self.sequence_map[seq][next_val] += 1
            
    def pred(self, input_seq):
        """Predict next token (-1, 0, 1) based on sequence"""
        seq_key = tuple(input_seq)
        if seq_key in self.sequence_map and self.sequence_map[seq_key]:
            prediction = self.sequence_map[seq_key].most_common(1)[0][0]
            return prediction
        else:
            return 0 # Default to Flat/No Trade if unknown

    def grid(self):
        print("\nStarting Grid Search for 'a' (Floor 1%)...")
        # Lowest bin limited to 1% (0.01)
        a_values = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
        best_score = -float('inf')
        
        for a in a_values:
            t_tokens, t_returns, _, _ = self.prepare(a)
            self.map_sequences(t_tokens)
            
            # Score metrics
            wins = 0
            losses = 0
            
            if len(t_tokens) < 6: continue
            
            for i in range(len(t_tokens) - 5):
                current_seq = t_tokens[i : i+5]
                # Note: We don't predict price, we predict direction (-1, 0, 1)
                predicted_token = self.pred(current_seq)
                
                # Check outcome against REAL return
                actual_ret = t_returns[i+5]
                
                # Logic: 
                # If we predict 1 (Up), we profit if actual_ret > 0
                # If we predict -1 (Down), we profit if actual_ret < 0
                
                trade_pnl = actual_ret * predicted_token
                
                # We only count "Wins" if the move was substantial enough to cover fees/slippage
                # Using 'a' as a proxy for significant win
                if trade_pnl > 0: 
                    wins += 1
                elif trade_pnl < 0:
                    losses += 1
            
            score = wins - losses
            print(f"a={a:.3f}: Score={score} (Wins {wins} / Losses {losses})")
            
            if score > best_score:
                best_score = score
                self.best_a = a
        
        print(f"Optimization complete. Best a: {self.best_a}")

    def test(self):
        print(f"\nRunning Test with best a={self.best_a}...")
        t_tokens, t_returns, test_tokens, test_returns = self.prepare(self.best_a)
        
        # Train on TRAIN data
        self.map_sequences(t_tokens)
        
        if len(test_tokens) < 6:
            print("Not enough test data.")
            return

        cumulative_pnl = [0]
        wins = 0
        losses = 0
        total_predictions = 0
        flat_predictions = 0
        
        for i in range(len(test_tokens) - 5):
            current_seq = test_tokens[i : i+5]
            predicted_token = self.pred(current_seq)
            total_predictions += 1
            
            if predicted_token == 0:
                flat_predictions += 1
                # Append previous PnL to keep graph continuity
                cumulative_pnl.append(cumulative_pnl[-1])
                continue

            actual_ret = test_returns[i+5]
            
            # PnL = Direction * Return
            # If pred=1 and ret=0.02 -> profit 0.02
            # If pred=-1 and ret=-0.02 -> profit 0.02
            trade_pnl = predicted_token * actual_ret
            
            cumulative_pnl.append(cumulative_pnl[-1] + trade_pnl)
            
            if trade_pnl > 0:
                wins += 1
            elif trade_pnl < 0:
                losses += 1
        
        denominator = wins + losses
        accuracy = 0
        if denominator > 0:
            accuracy = wins / denominator
            
        print(f"Results on Test Data: Accuracy={accuracy:.2%}, Wins={wins}, Losses={losses}")
        print(f"Total Predictions: {total_predictions}")
        print(f"Flat Predictions (No Trade): {flat_predictions}")
        print(f"Final Cumulative Return: {cumulative_pnl[-1]:.4f} ({(cumulative_pnl[-1]*100):.2f}%)")
        
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_pnl, label='Cumulative Returns', color='blue')
        plt.title(f'Strategy Returns (a={self.best_a}) | Acc: {accuracy:.2%}')
        plt.xlabel('Hours (Test Set)')
        plt.ylabel('Return (1.0 = 100%)')
        plt.legend()
        plt.grid(True)
        
        self.upload(plt, "plot/pnl.png")
        plt.close()

    def main(self):
        self.fetch()
        if self.df is None or len(self.df) == 0: return

        print("\nGenerating plot of all data...")
        # Just plot the raw close prices for context
        plt.figure(figsize=(15, 7))
        plt.plot(self.df['close'].values, label='ETH Price', linewidth=0.5)
        plt.axvline(x=len(self.df)//2, color='r', linestyle='--', label='Train/Test Split')
        plt.title(f"All ETH/USDT 1h Data")
        plt.legend()
        plt.grid(True)
        
        self.upload(plt, "plot/plot.png")
        plt.close()
        
        self.grid()
        self.test()
        print("\nExit.")

if __name__ == "__main__":
    model = ETHModel()
    model.main()
