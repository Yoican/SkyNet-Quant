import requests
import pandas as pd
import json
import os
import time

def get_tx_history(symbol):
    prefix = 'sh' if symbol.startswith(('5', '6')) else 'sz'
    var_name = "kline_dayqfq"
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?_var={var_name}&param={prefix}{symbol},day,,,500,qfq"
    
    try:
        r = requests.get(url, timeout=10)
        text = r.text
        # 去掉 JS 变量名前缀：kline_dayqfq=
        json_str = text[len(var_name)+1:]
        data = json.loads(json_str)
        
        target_data = data['data'][f"{prefix}{symbol}"]
        kline = target_data['qfqday'] if 'qfqday' in target_data else target_data['day']
        
        df = pd.DataFrame(kline)
        df = df.iloc[:, 0:6]
        df.columns = ['date', 'open', 'close', 'high', 'low', 'vol']
        return df
    except Exception as e:
        print(f"TX Error: {str(e)}")
        return None

def tx_full_sync():
    print("[TX FULL SYNC] OHLCV 500-Day History Update (JS Fix)...")
    UNIVERSE_FILE = 'skynet_core/skynet_universe.json'
    HISTORY_DIR = 'skynet_core/history_data/'
    
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)

    with open(UNIVERSE_FILE, 'r', encoding='utf-8') as f:
        universe = json.load(f)
    
    for sym in universe['assets'].keys():
        print(f"Syncing {sym}...", end=" ", flush=True)
        df = get_tx_history(sym)
        if df is not None:
            df.to_csv(f"{HISTORY_DIR}{sym}.csv", index=False)
            print(f"OK ({len(df)} days)")
        else:
            print("FAILED")
        # Slow sync to avoid IP ban
        time.sleep(0.5)

if __name__ == "__main__":
    tx_full_sync()
