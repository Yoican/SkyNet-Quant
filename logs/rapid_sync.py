import requests
import pandas as pd
import json
import os
import time
import concurrent.futures
import sys
import datetime

def get_tx_price_worker(symbol):
    prefix = 'sh' if symbol.startswith(('5', '6')) else 'sz'
    url = f"https://qt.gtimg.cn/q={prefix}{symbol}"
    try:
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            parts = r.text.split('~')
            if len(parts) > 3:
                return {
                    "symbol": symbol,
                    "name": parts[1],
                    "open": float(parts[5]),
                    "close": float(parts[3]),
                    "high": float(parts[33]),
                    "low": float(parts[34]),
                    "vol": float(parts[6]) * 100,
                    "status": "OK"
                }
    except:
        pass
    return {"symbol": symbol, "status": "FAILED"}

def fast_parallel_sync():
    print("[SkyNet Rapid Sync] Starting parallel fetch...")
    start_time = time.time()
    
    UNIVERSE_FILE = 'skynet_universe.json'
    with open(UNIVERSE_FILE, 'r', encoding='utf-8') as f:
        universe = json.load(f)
    symbols = list(universe['assets'].keys())

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(get_tx_price_worker, symbols))

    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    success_count = 0
    daily_prices = {}
    for res in results:
        if res['status'] == 'OK':
            daily_prices[res['symbol']] = {
                "name": res['name'],
                "close": res['close'],
                "date": today_str
            }
            csv_path = f"history_data/{res['symbol']}.csv"
            
            if not os.path.exists(csv_path):
                # 如果文件不存在，创建一个带标题的空 CSV
                with open(csv_path, 'w', encoding='utf-8') as f:
                    f.write("date,open,close,high,low,vol\n")
            
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # 强制填充 0 直到与 df 的列数一致
                num_cols = len(df.columns)
                new_row = [today_str, res['open'], res['close'], res['high'], res['low'], res['vol']]
                while len(new_row) < num_cols:
                    new_row.append(0)
                
                # 修改日期比较逻辑，确保字符串比较正确
                last_date = str(df.iloc[-1, 0]) if not df.empty else ""
                if not df.empty and last_date == today_str:
                    df.iloc[-1] = new_row
                else:
                    df.loc[len(df)] = new_row
                df.to_csv(csv_path, index=False)
                success_count += 1
    
    # 同时更新 skynet_daily_prices.json
    PRICE_FILE = 'skynet_daily_prices.json'
    with open(PRICE_FILE, 'w', encoding='utf-8') as f:
        json.dump(daily_prices, f, ensure_ascii=False, indent=4)
    
    end_time = time.time()
    print(f"Sync Complete. Success: {success_count}/{len(symbols)}. Time: {end_time - start_time:.2f}s")

if __name__ == "__main__":
    fast_parallel_sync()
