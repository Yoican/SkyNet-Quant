import pandas as pd
import numpy as np
import json
import os
import sys

# 设置编码
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_DIR = os.path.join(BASE_DIR, 'history_data')
UNIVERSE_FILE = os.path.join(BASE_DIR, 'skynet_universe.json')

def get_latest_momentum():
    with open(UNIVERSE_FILE, 'r', encoding='utf-8') as f:
        universe = json.load(f)
    symbols = list(universe['assets'].keys())
    
    momentum_list = []
    for sym in symbols:
        csv_path = os.path.join(HISTORY_DIR, f"{sym}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if len(df) > 20:
                # 计算近5日涨幅作为简单预期收益代理
                df['pct_chg_5d'] = df['close'].pct_change(5)
                last_momentum = df['pct_chg_5d'].iloc[-1]
                name = universe['assets'][sym]['name']
                momentum_list.append({
                    "symbol": sym,
                    "name": name,
                    "momentum": last_momentum,
                    "price": df['close'].iloc[-1]
                })
    
    # 按动量排序
    momentum_list.sort(key=lambda x: x['momentum'], reverse=True)
    return momentum_list

if __name__ == "__main__":
    results = get_latest_momentum()
    print(json.dumps(results, ensure_ascii=False, indent=2))
