import akshare as ak
import pandas as pd
import json
import os
import time

def full_history_sync():
    print("[SkyNet Full Sync] OHLCV History Update Start (TX Mode)...")
    
    UNIVERSE_FILE = 'skynet_core/skynet_universe.json'
    HISTORY_DIR = 'skynet_core/history_data/'
    
    if not os.path.exists(UNIVERSE_FILE):
        return
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)

    with open(UNIVERSE_FILE, 'r', encoding='utf-8') as f:
        universe = json.load(f)
    
    symbols = list(universe['assets'].keys())
    
    for sym in symbols:
        print(f"Syncing OHLCV for {sym}...", end=" ", flush=True)
        try:
            # 获取历史全量数据 (为了盲测，需要 OHLCV)
            # TX 接口虽然稳，但 akshare 的 em 接口数据更全。
            # 如果 EM 挂了，我们这里的 fetch 逻辑要更顽强。
            df = ak.fund_etf_hist_em(symbol=sym, period="daily", start_date="20200101", end_date="20260318", adjust="qfq")
            if not df.empty:
                df.columns = ['date', 'open', 'close', 'high', 'low', 'vol', 'amt', 'amp', 'pct', 'chg', 'turnover']
                df.to_csv(f"{HISTORY_DIR}{sym}.csv", index=False)
                print(f"OK ({len(df)} rows)")
            else:
                print("EMPTY")
        except Exception as e:
            print(f"FAILED: {str(e)}")
        time.sleep(1)

if __name__ == "__main__":
    full_history_sync()
