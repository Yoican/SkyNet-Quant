import pandas as pd
import json
import os
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_DIR = os.path.join(BASE_DIR, 'history_data')
PRICE_FILE = os.path.join(BASE_DIR, 'skynet_daily_prices.json')
UNIVERSE_FILE = os.path.join(BASE_DIR, 'skynet_universe.json')

def sync_prices_from_csv():
    with open(UNIVERSE_FILE, 'r', encoding='utf-8') as f:
        universe = json.load(f)
    
    symbols = list(universe['assets'].keys())
    prices = {}
    today_str = datetime.datetime.now().strftime('%Y-%m-%d')
    
    for sym in symbols:
        csv_path = os.path.join(HISTORY_DIR, f"{sym}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if not df.empty:
                last_row = df.iloc[-1]
                # Assuming CSV columns are [date, open, high, low, close, volume]
                # If they have headers, we use labels. If not, use indices.
                try:
                    prices[sym] = {
                        "name": universe['assets'].get(sym, {}).get('name', sym),
                        "date": str(last_row.iloc[0]),
                        "close": float(last_row.iloc[4]),
                        "status": "HEALTHY"
                    }
                except Exception as e:
                    print(f"Error parsing {sym}: {e}")
                    
    with open(PRICE_FILE, 'w', encoding='utf-8') as f:
        json.dump(prices, f, indent=4, ensure_ascii=False)
    print(f"Synced {len(prices)} symbols from CSV to skynet_daily_prices.json")

if __name__ == "__main__":
    sync_prices_from_csv()
