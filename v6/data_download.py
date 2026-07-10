import os
import time
import pandas as pd
from datetime import datetime, timedelta
from dukascopy_python import fetch as get_price
import dukascopy_python

OUTPUT_FILE = r"C:\Users\Archit Gore\Desktop\indicator based algo trading\v2\data.csv"
INSTRUMENT = "XAU/USD"
CHUNK_DAYS = 7   # fetch 1 week at a time
OFFER_SIDE = "mid"  # or "ask", "mid"
MAX_RETRIES = float("inf")  # retry forever on net issues

def safe_download(start, end):
    """Download a chunk of data with retries"""
    attempt = 0
    while True:
        try:
            df = get_price(
                instrument=INSTRUMENT,
                start=start,
                end=end,
                interval=dukascopy_python.INTERVAL_MIN_1,     # 1-minute candles
                offer_side=OFFER_SIDE
            )
            if df is None or df.empty:
                print(f"No data for {start} -> {end}")
                return None
            return df
        except Exception as e:
            attempt += 1
            print(f"Error: {e}, retrying in 10s (attempt {attempt})...")
            time.sleep(10)

def load_existing():
    """Load existing CSV if available"""
    if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
        return pd.read_csv(OUTPUT_FILE, parse_dates=["timestamp"])
    return pd.DataFrame()

def save_append(df):
    """Append new rows to CSV"""
    header = not os.path.exists(OUTPUT_FILE) or os.path.getsize(OUTPUT_FILE) == 0
    df.to_csv(OUTPUT_FILE, mode="a", header=header, index=True)

def main():
    existing = load_existing()

    # If no file exists yet, start from now
    if existing.empty:
        current_end = datetime.now()
    else:
        # Resume from the earliest already-downloaded timestamp
        current_end = existing['timestamp'].min().to_pydatetime()

    print(f"Starting download backwards from: {current_end}")

    while True:
        current_start = current_end - timedelta(days=CHUNK_DAYS)
        print(f"Fetching {current_start} -> {current_end}...")

        df = safe_download(current_start, current_end)
        if df is None:
            break

        # Ensure chronological order
        df = df.sort_index()

        # Avoid duplicates if resuming
        if not existing.empty:
            new_times = set(existing.index)
            df = df[~df.index.isin(new_times)]

        if not df.empty:
            save_append(df.iloc[::-1])
            print(f"Saved {len(df)} rows ({df.index.min()} -> {df.index.max()})")
        else:
            print("No new rows, stopping.")
            break

        # Move window backwards
        current_end = current_start

if __name__ == "__main__":
    main()
