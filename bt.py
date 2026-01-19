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
        self.github_path = "plot.png"  # Added extension for validity

    def fetch(self):
        """Fetch ohlc eth 1h binance all data"""
        print("Fetching ETH/USDT 1h data from Binance...")
        exchange = ccxt.binance()
        symbol = 'ETH/USDT'
        timeframe = '1h'
        
        # Binance limits to 1000 candles, need to paginate backwards
        # Starting from 'now'
        since = exchange.parse8601('2017-08-17T00:00:00Z') # Approx ETH listing
        all_ohlcv = []
        
        while True:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
                if len(ohlcv) == 0:
                    break
                all_ohlcv.extend(ohlcv)
                # Update 'since' to the last timestamp + 1h in ms
                since = ohlcv[-1][0] + 60 * 60 * 1000 
                
                # Check if we reached current time
                if since > exchange.milliseconds():
                    break
                    
                # Rate limit safety
                time.sleep(exchange.rateLimit / 1000)
                print(f"Fetched {len(all_ohlcv)} candles...", end='\r')
            except Exception as e:
                print(f"\nError fetching data: {e}")
                break
                
        self.df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        print(f"\nTotal data fetched: {len(self.df)} rows.")

    def prepare(self, a=None):
        """
        Keep close
        Normalize to first close
        Round to floor where a is specified (default 0.01)
        Split 80/20
        """
        if a is None:
            a = self.best_a
            
        # Keep close
        prices = self.df['close'].values.astype(float)
        
        # Normalize to first close
        normalized = prices / prices[0]
        
        # Round to floor where a is step
        # formula: floor(x / a) * a
        discretized = np.floor(normalized / a) * a
        
        # Split 80/20
        split_idx = int(len(discretized) * 0.8)
        self.train_data = discretized[:split_idx]
        self.test_data = discretized[split_idx:]
        
        return self.train_data, self.test_data

    def upload(self, plt_obj, filename="plot.png"):
        """Save plot to GitHub via API"""
        token = os.getenv('PAT')
        if not token:
            print("Error: PAT not found in .env")
            return

        # Save plot to buffer
        buf = io.BytesIO()
        plt_obj.savefig(buf, format='png')
        buf.seek(0)
        content_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        url = f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}/contents/{filename}"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        # Check if file exists to get SHA (needed for update)
        get_resp = requests.get(url, headers=headers)
        data = {
            "message": f"Update plot for a={self.best_a}",
            "content": content_base64
        }
        
        if get_resp.status_code == 200:
            data["sha"] = get_resp.json()['sha']
            
        resp = requests.put(url, json=data, headers=headers)
        if resp.status_code in [200, 201]:
            print(f"Successfully uploaded {filename} to GitHub.")
        else:
            print(f"Failed to upload: {resp.status_code} {resp.text}")

    def map_sequences(self, data):
        """Map all sequences of 6 with their probabilities"""
        self.sequence_map.clear()
        
        # Sliding window of size 6
        # Sequence: [t-5, t-4, t-3, t-2, t-1, t]
        # Key: tuple(first 5), Value: Counter(6th)
        for i in range(len(data) - 6):
            seq = tuple(data[i : i+5])
            next_val = data[i+5]
            self.sequence_map[seq][next_val] += 1
            
    def pred(self, input_seq):
        """
        Take input 5
        Output 6 starting with 5 with highest probability
        Returns: The predicted 6th value (next step)
        """
        seq_key = tuple(input_seq)
        if seq_key in self.sequence_map:
            # Get the value with highest count
            prediction = self.sequence_map[seq_key].most_common(1)[0][0]
            return prediction
        else:
            # Fallback: return last value (no change) if sequence unseen
            return input_seq[-1]

    def grid(self):
        """
        Grid search a to optimize times pnl > a - times pnl < -a
        """
        print("Starting Grid Search for 'a'...")
        # Search range for 'a' (e.g., 0.005 to 0.05)
        a_values = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
        best_score = -float('inf')
        
        for a in a_values:
            train, _ = self.prepare(a)
            self.map_sequences(train)
            
            pnl_greater_a = 0
            pnl_less_neg_a = 0
            
            # Evaluate on training set (last part of it to save time, or full)
            # We simulate trading based on predictions
            for i in range(len(train) - 6):
                current_seq = train[i : i+5]
                current_price = train[i+4] # The last known price
                actual_next = train[i+5]
                
                predicted_next = self.pred(current_seq)
                
                # Logic: If pred > current, Buy. If pred < current, Sell.
                # PnL = (Actual Next - Current) * Direction
                
                direction = 0
                if predicted_next > current_price:
                    direction = 1
                elif predicted_next < current_price:
                    direction = -1
                
                # Calculate PnL magnitude
                # Note: prices are normalized, so difference is percentage-like
                trade_pnl = (actual_next - current_price) * direction
                
                if trade_pnl > a:
                    pnl_greater_a += 1
                elif trade_pnl < -a:
                    pnl_less_neg_a += 1
            
            score = pnl_greater_a - pnl_less_neg_a
            print(f"a={a}: Score={score} (+{pnl_greater_a} / -{pnl_less_neg_a})")
            
            if score > best_score:
                best_score = score
                self.best_a = a
        
        print(f"Optimization complete. Best a: {self.best_a}")

    def test(self):
        """
        Test on test data
        Plot pnl 
        Calculate accuracy
        """
        print("Running Test on unseen data...")
        # Prepare data with optimal a
        _, test = self.prepare(self.best_a)
        
        # Need to map on train data again to ensure model is ready with best a
        train_final, _ = self.prepare(self.best_a)
        self.map_sequences(train_final)
        
        pnls = []
        count_pnl_gt_a = 0
        count_pnl_lt_neg_a = 0
        cumulative_pnl = [0]
        
        for i in range(len(test) - 6):
            current_seq = test[i : i+5]
            current_price = test[i+4]
            actual_next = test[i+5]
            
            predicted_next = self.pred(current_seq)
            
            direction = 0
            if predicted_next > current_price:
                direction = 1
            elif predicted_next < current_price:
                direction = -1
                
            trade_pnl = (actual_next - current_price) * direction
            pnls.append(trade_pnl)
            cumulative_pnl.append(cumulative_pnl[-1] + trade_pnl)
            
            if trade_pnl > self.best_a:
                count_pnl_gt_a += 1
            elif trade_pnl < -self.best_a:
                count_pnl_lt_neg_a += 1
        
        # Accuracy Metric: times pnl > a / (times pnl > a + times pnl < -a)
        denominator = count_pnl_gt_a + count_pnl_lt_neg_a
        accuracy = 0
        if denominator > 0:
            accuracy = count_pnl_gt_a / denominator
            
        print(f"Accuracy (Significant Moves): {accuracy:.2%}")
        print(f"Total Significant Wins: {count_pnl_gt_a}")
        print(f"Total Significant Losses: {count_pnl_lt_neg_a}")
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_pnl, label='Cumulative PnL (Normalized Units)')
        plt.title(f'Strategy PnL (a={self.best_a}) | Acc: {accuracy:.2%}')
        plt.xlabel('Trades')
        plt.ylabel('PnL')
        plt.legend()
        plt.grid(True)
        
        # Save and Upload
        self.upload(plt, "pnl_plot.png")
        plt.close()

    def main(self):
        self.fetch()
        
        # Initial Plot of data (optional, but requested in step 4 implies general plot)
        # We will plot the raw normalized training data first
        train, _ = self.prepare(0.01) # Temp prepare for plot
        plt.figure(figsize=(12, 6))
        plt.plot(train[:500]) # Plot first 500 points
        plt.title("Normalized & Floored ETH Data (Snippet)")
        self.upload(plt, "data_plot.png")
        plt.close()
        
        # Run optimization
        self.grid()
        
        # Run Test
        self.test()
        
        print("Exit.")

if __name__ == "__main__":
    model = ETHModel()
    model.main()
