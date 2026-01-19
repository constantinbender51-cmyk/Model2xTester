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
        self.train_data = [] # Will store pct changes
        self.test_data = []  # Will store pct changes
        self.raw_prices = [] # Store raw prices to calculate PnL later
        self.sequence_map = defaultdict(Counter)
        self.best_a = 0.001 # Default step for percentage (0.1%)
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
        """
        Convert prices to Percentage Changes -> Round to 'a' floor -> Split
        """
        if self.df is None or len(self.df) == 0:
            print("No data to prepare.")
            return [], []

        if a is None:
            a = self.best_a
            
        prices = self.df['close'].values.astype(float)
        self.raw_prices = prices # Save for PnL calculation
        
        # Calculate Percentage Change: (Price_t - Price_t-1) / Price_t-1
        # Insert 0 at the start to keep length consistent with prices
        pct_changes = np.diff(prices, prepend=prices[0]) / prices
        # Fix the first element (prepend) being 0
        pct_changes[0] = 0 
        
        # Round to floor where a is step (e.g. 0.001 = 0.1%)
        # Adding epsilon to handle floating point noise
        discretized_pct = np.floor((pct_changes + 1e-9) / a) * a
        
        # Split 50/50
        split_idx = int(len(discretized_pct) * 0.5)
        self.train_data = discretized_pct[:split_idx]
        self.test_data = discretized_pct[split_idx:]
        
        # Also split raw prices for testing accuracy later
        self.train_prices = prices[:split_idx]
        self.test_prices = prices[split_idx:]
        
        return self.train_data, self.test_data

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

    def map_sequences(self, data):
        """Map sequences of 5 percentage changes to the 6th"""
        self.sequence_map.clear()
        if len(data) < 6: return

        # Sequence: [pct_1, pct_2, pct_3, pct_4, pct_5] -> predicts pct_6
        for i in range(len(data) - 5):
            seq = tuple(data[i : i+5])
            next_val = data[i+5]
            self.sequence_map[seq][next_val] += 1
            
    def pred(self, input_seq):
        """Returns the most probable next PERCENTAGE move"""
        seq_key = tuple(input_seq)
        if seq_key in self.sequence_map and self.sequence_map[seq_key]:
            prediction = self.sequence_map[seq_key].most_common(1)[0][0]
            return prediction
        else:
            # Fallback: Assume 0% change (flat) if unknown sequence
            return 0.0

    def grid(self):
        """
        Optimize 'a' (percentage bucket size).
        Range: 0.0005 (0.05%) to 0.005 (0.5%)
        """
        print("\nStarting Grid Search for 'a' (Percentage Change buckets)...")
        a_values = [0.0005, 0.001, 0.002, 0.003, 0.004, 0.005]
        best_score = -float('inf')
        
        for a in a_values:
            train_pcts, _ = self.prepare(a)
            self.map_sequences(train_pcts)
            
            pnl_greater_a = 0
            pnl_less_neg_a = 0
            
            if len(train_pcts) < 6: continue
            
            # Use raw prices for accurate PnL calculation
            prices_subset = self.train_prices
            
            for i in range(len(train_pcts) - 5):
                current_seq = train_pcts[i : i+5]
                current_price = prices_subset[i+4]
                actual_next_price = prices_subset[i+5]
                
                # Predict the next % move
                pred_pct = self.pred(current_seq)
                
                # Direction is based on the SIGN of the predicted % change
                direction = 0
                if pred_pct > 0:
                    direction = 1
                elif pred_pct < 0:
                    direction = -1
                
                # PnL = (Actual Price Move) * Direction
                trade_pnl = (actual_next_price - current_price) * direction
                
                # We normalize PnL by price to keep score consistent across years
                # Normalized PnL = trade_pnl / current_price
                norm_pnl = trade_pnl / current_price
                
                if norm_pnl > a:
                    pnl_greater_a += 1
                elif norm_pnl < -a:
                    pnl_less_neg_a += 1
            
            score = pnl_greater_a - pnl_less_neg_a
            print(f"a={a:.4f}: Score={score} (+{pnl_greater_a} / -{pnl_less_neg_a})")
            
            if score > best_score:
                best_score = score
                self.best_a = a
        
        print(f"Optimization complete. Best a: {self.best_a}")

    def test(self):
        print(f"\nRunning Test with best a={self.best_a}...")
        train_pcts, test_pcts = self.prepare(self.best_a)
        
        # Map using training data
        self.map_sequences(train_pcts)
        
        if len(test_pcts) < 6:
            print("Not enough test data.")
            return

        pnls = []
        count_pnl_gt_a = 0
        count_pnl_lt_neg_a = 0
        cumulative_pnl = [0]
        
        total_predictions = 0
        flat_predictions = 0
        
        prices_subset = self.test_prices
        
        # Loop through TEST data
        for i in range(len(test_pcts) - 5):
            current_seq = test_pcts[i : i+5]
            
            # Get actual prices for PnL (Offset by 5 because sequences consume 5 steps)
            # index i in test_pcts corresponds to predicting move at i+5
            current_price = prices_subset[i+4]
            actual_next_price = prices_subset[i+5]
            
            # Prediction is a PERCENTAGE move
            pred_pct = self.pred(current_seq)
            total_predictions += 1
            
            direction = 0
            if pred_pct > 0:
                direction = 1
            elif pred_pct < 0:
                direction = -1
            else:
                flat_predictions += 1
                
            # Trade PnL ($)
            trade_pnl_usd = (actual_next_price - current_price) * direction
            
            # Normalize PnL to units of 'a' for the chart (so 2017 and 2022 are comparable)
            # This shows "Percent Return" accumulation
            pct_return = trade_pnl_usd / current_price
            
            pnls.append(pct_return)
            cumulative_pnl.append(cumulative_pnl[-1] + pct_return)
            
            if pct_return > self.best_a:
                count_pnl_gt_a += 1
            elif pct_return < -self.best_a:
                count_pnl_lt_neg_a += 1
        
        denominator = count_pnl_gt_a + count_pnl_lt_neg_a
        accuracy = 0
        if denominator > 0:
            accuracy = count_pnl_gt_a / denominator
            
        print(f"Results on Test Data: Accuracy={accuracy:.2%}, Wins={count_pnl_gt_a}, Losses={count_pnl_lt_neg_a}")
        print(f"Total Predictions: {total_predictions}")
        print(f"Flat Predictions (No Trade): {flat_predictions}")
        
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_pnl, label='Cumulative Return %', color='green')
        plt.title(f'Strategy PnL (Relative % Mode, a={self.best_a}) | Acc: {accuracy:.2%}')
        plt.xlabel('Trades')
        plt.ylabel('Cumulative Return (1.0 = 100%)')
        plt.legend()
        plt.grid(True)
        
        self.upload(plt, "plot/pnl.png")
        plt.close()

    def main(self):
        self.fetch()
        if self.df is None or len(self.df) == 0: return

        print("\nGenerating plot of all data...")
        # Visualizing Percentage Volatility instead of normalized price
        viz_a = 0.005
        train_viz, test_viz = self.prepare(viz_a)
        all_data = np.concatenate((train_viz, test_viz))
        split_index = len(train_viz)
        
        plt.figure(figsize=(15, 7))
        # Plotting a subset of volatility to avoid clutter, or a rolling average
        # Let's plot the "discretized percent change" to show what the model sees
        plt.plot(all_data, label=f'Discretized % Change (a={viz_a})', linewidth=0.1, alpha=0.7)
        plt.axvline(x=split_index, color='r', linestyle='--', linewidth=1.5, label='Split (50/50)')
        plt.title(f"Market 'Heartbeat': % Change Sequences (N={len(all_data)})")
        plt.legend()
        plt.ylim(-0.05, 0.05) # Limit view to +/- 5% candles for clarity
        plt.grid(True)
        
        self.upload(plt, "plot/plot.png")
        plt.close()
        
        self.grid()
        self.test()
        print("\nExit.")

if __name__ == "__main__":
    model = ETHModel()
    model.main()
