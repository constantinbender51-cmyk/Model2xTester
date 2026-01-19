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
GITHUB_PAT = os.getenv("GITHUB_PAT")

# 2. Configuration
BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "ETHUSDT"
INTERVAL = "1m"
START_DATE = "2022-01-01"  # Updated start date
END_DATE = "2026-01-01"

# GitHub Config
REPO_OWNER = "constantinbender51-cmyk"
REPO_NAME = "Models"
REMOTE_FILE_PATH = "ohlc/eth1mohlc.csv"
BRANCH = "main"
GITHUB_API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{REMOTE_FILE_PATH}"

# Local Fallback Config
LOCAL_FILE_PATH = "/app/data/ethohlc1m.csv"

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
            last_close_time = klines[-1][6]
            current_start = last_close_time + 1
            
            # Progress log
            if len(data) % 100000 == 0:
                last_date = datetime.fromtimestamp(klines[-1][0]/1000, timezone.utc).strftime('%Y-%m-%d')
                print(f"Fetched up to {last_date} | Rows: {len(data)}")
            
            time.sleep(0.05) 
            
        except Exception as e:
            print(f"\nError fetching data: {e}")
            break

    print(f"\nDownload complete. Total rows: {len(data)}")
    
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume", 
        "close_time", "qav", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    
    # Filter and Format
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    cols = ["open", "high", "low", "close", "volume"]
    df[cols] = df[cols].astype(float)
    df["volume"] = df["volume"].round(3)
    
    return df

def save_local(df, path):
    """Saves DataFrame to local path, creating directories if needed."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"SUCCESS: Saved locally to {path}")
    except Exception as e:
        print(f"CRITICAL: Failed to save locally. {e}")

def upload_to_github(df, token, url, branch):
    """Returns True if successful, False if failed."""
    if not token:
        print("No GITHUB_PAT found. Skipping upload.")
        return False

    print("Preparing GitHub upload...")
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    # Check size (Limit ~100MB)
    size_mb = len(csv_content.encode('utf-8')) / (1024 * 1024)
    print(f"Payload Size: {size_mb:.2f} MB")
    
    if size_mb > 99:
        print("Upload rejected: File exceeds GitHub 100MB API limit.")
        return False

    try:
        # Check for existing file sha
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"}
        get_resp = requests.get(url, headers=headers)
        sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None

        # Upload
        content_encoded = base64.b64encode(csv_content.encode('utf-8')).decode('utf-8')
        data = {
            "message": f"Update {SYMBOL} OHLC {START_DATE}-{END_DATE}",
            "content": content_encoded,
            "branch": branch
        }
        if sha: data["sha"] = sha
            
        response = requests.put(url, headers=headers, json=data)
        
        if response.status_code in [200, 201]:
            print("SUCCESS: Uploaded to GitHub.")
            return True
        else:
            print(f"GitHub Upload Failed: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        print(f"GitHub Upload Exception: {e}")
        return False

if __name__ == "__main__":
    df = fetch_binance_data(SYMBOL, INTERVAL, START_DATE, END_DATE)
    
    if not df.empty:
        # Try GitHub Upload
        success = upload_to_github(df, GITHUB_PAT, GITHUB_API_URL, BRANCH)
        
        # Fallback if upload fails
        if not success:
            print("Initiating fallback save...")
            save_local(df, LOCAL_FILE_PATH)
    else:
        print("No data fetched.")
