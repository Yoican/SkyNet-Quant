import pandas as pd
import numpy as np
import json
import os
import sys
import math

# 设置编码
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_DIR = os.path.join(BASE_DIR, 'history_data')
UNIVERSE_FILE = os.path.join(BASE_DIR, 'skynet_universe.json')

def analyze_best_setup():
    with open(UNIVERSE_FILE, 'r', encoding='utf-8') as f:
        universe = json.load(f)
    symbols = list(universe['assets'].keys())

    momentum_data = []
    for sym in symbols:
        csv_path = os.path.join(HISTORY_DIR, f"{sym}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # 兼容新加入的 CSV (数据行数少于 60 的情况)
            if len(df) > 1:
                # 如果没有 60 天数据，取现有的最大窗口
                window = min(len(df)-1, 5)
                pct_5d = df['close'].pct_change(window).iloc[-1] if window > 0 else 0
                vol = df['close'].pct_change().std() * math.sqrt(252)
                sharpe = pct_5d / vol if (vol > 0 and not pd.isna(vol)) else 0
                momentum_data.append({
                    "symbol": sym,
                    "name": universe['assets'][sym]['name'],
                    "momentum_5d": pct_5d,
                    "vol": vol,
                    "sharpe": sharpe,
                    "price": df['close'].iloc[-1]
                })

    momentum_data.sort(key=lambda x: x['sharpe'], reverse=True)
    
    # 获取夏普比率排名前 8 的资产作为“弩”
    top_8_sharpe = momentum_data[:8]
    
    # 动态构建权重：
    # 1. 黄金 (518880) - 20% [盾]
    # 2. 红利 (510880) - 15% [盾]
    # 3. 排名前 8 的高夏普资产，平分剩余的 65% (每支约 8.1%)
    
    proposed_weights = {
        "518880": 0.20,
        "510880": 0.15
    }
    
    per_asset_weight = 0.65 / 8
    for asset in top_8_sharpe:
        if asset['symbol'] not in proposed_weights:
            proposed_weights[asset['symbol']] = per_asset_weight

    return proposed_weights, top_8_sharpe

    # 为了让 skynet_live 脚本能读到 top_assets
    top_assets = [a for a in momentum_data if a['symbol'] in proposed_weights]

    return proposed_weights, top_assets

if __name__ == "__main__":
    weights, assets = analyze_best_setup()
    print("\n--- 🛡️ SkyNet '盾+弩' 初始配置方案 ---")
    print("\n[目标权重清单]:")
    for s, w in weights.items():
        print(f"- {s}: {w:.2%}")
