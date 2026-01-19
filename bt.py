import os
import time
import requests
import pandas as pd
import io
import base64
from datetime import datetime, timezone
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()
GITHUB_PAT = os.getenv("PAT")  # Ensure your .env key is named GITHUB_PAT or update this

if not GITHUB_PAT:
    raise ValueError("Error: GITHUB_PAT not found in .env file.")

# 2. Configuration
BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "ETHUSDT"
INTERVAL = "1m"
START_DATE = "2020-01-01"
END_DATE = "2026-01-01"

# GitHub Config
REPO_OWNER = "constantinbender51-cmyk"
REPO_NAME = "Models"
FILE_PATH = "ohlc/eth1mohlc.csv"
BRANCH = "main"
GITHUB_API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"

def get_timestamp(date_str):
    return int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

def fetch_binance_data(symbol, interval, start_str, end_str):
    start_ts = get_timestamp(start_str)
    end_ts = get_timestamp(end_str)
    
    data = []
    current_start = start_ts
    
    print(f"Fetching {symbol} {interval} data from {start_str} to {end_str}...")
    
    while current_start < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ts,
            "limit": 1000
        }
        
        try:
            response = requests.get(BINANCE_URL, params=params)
            response.raise_for_status()
            klines = response.json()
            
            if not klines:
                break
            
            data.extend(klines)
            
            # Update start time to the close time of the last candle + 1ms
            last_close_time = klines[-1][6]
            current_start = last_close_time + 1
            
            # Progress indicator
            last_date = datetime.fromtimestamp(klines[-1][0]/1000, timezone.utc).strftime('%Y-%m-%d')
            print(f"\rFetched up to {last_date} | Total rows: {len(data)}", end="", flush=True)
            
            # Rate limit handling (Binance is generous, but safe side)
            time.sleep(0.1)
            
        except Exception as e:
            print(f"\nError fetching data: {e}")
            break

    print(f"\nDownload complete. Total rows: {len(data)}")
    
    # Create DataFrame
    # Binance columns: Open Time, Open, High, Low, Close, Volume, Close Time, Quote Asset Vol, Trades, Taker Buy Base, Taker Buy Quote, Ignore
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume", 
        "close_time", "qav", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    
    # Keep only relevant OHLCV columns to save space
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    
    # Convert types for optimization
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    cols = ["open", "high", "low", "close", "volume"]
    df[cols] = df[cols].astype(float)
    
    # Round volume to 3 decimals to reduce CSV string size
    df["volume"] = df["volume"].round(3)
    
    return df

def upload_to_github(df, token, url, branch):
    # Convert DF to CSV string
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    # Check size (GitHub Limit is 100MB)
    size_mb = len(csv_content.encode('utf-8')) / (1024 * 1024)
    print(f"CSV Size: {size_mb:.2f} MB")
    
    if size_mb > 99:
        print("WARNING: File exceeds GitHub API 100MB limit. Upload will likely fail.")
        print("Suggestion: Split the data by year.")
        return

    # Encode to Base64
    content_encoded = base64.b64encode(csv_content.encode('utf-8')).decode('utf-8')
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Check if file exists to get SHA (needed for update)
    get_resp = requests.get(url, headers=headers)
    sha = None
    if get_resp.status_code == 200:
        sha = get_resp.json().get("sha")
        print("File exists. Updating...")
    else:
        print("File does not exist. Creating...")

    data = {
        "message": f"Update {SYMBOL} OHLC data {START_DATE} to {END_DATE}",
        "content": content_encoded,
        "branch": branch
    }
    
    if sha:
        data["sha"] = sha
        
    response = requests.put(url, headers=headers, json=data)
    
    if response.status_code in [200, 201]:
        print("Successfully uploaded to GitHub.")
    else:
        print(f"Failed to upload. Status: {response.status_code}")
        print(response.json())

if __name__ == "__main__":
    df = fetch_binance_data(SYMBOL, INTERVAL, START_DATE, END_DATE)
    if not df.empty:
        upload_to_github(df, GITHUB_PAT, GITHUB_API_URL, BRANCH)
    else:
        print("No data fetched.")
