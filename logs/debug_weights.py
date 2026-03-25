import json
import pandas as pd
import os
import sys

# 关键：确保能导入 v11_modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from skynet_live_v11 import load_live_context, fetch_and_prep_daily_data

def debug_weights():
    print("🐥 [SkyNet Weight Debug] Analyzing Target Weights...")
    
    portfolio, prices = load_live_context()
    with open('skynet_universe.json', 'r', encoding='utf-8') as f:
        universe = json.load(f)
    
    symbols = list(universe['assets'].keys())
    
    # 模拟 V11 核心权重的简单分配逻辑 (复刻 skynet_live_v11.py 内部逻辑)
    # 风险资产分配
    risk_assets = [s for s in symbols if s != "518880"]
    gold_sym = "518880"
    
    target_weights = {}
    gold_weight = 0.40 # 基础防守
    target_weights[gold_sym] = gold_weight
    
    rem_weight = 1.0 - gold_weight
    for s in risk_assets:
        target_weights[s] = rem_weight / len(risk_assets)

    total_cap = portfolio['financials']['total_value']
    current_shares = portfolio['positions_in_shares']

    print(f"\n💰 Total Capital: {total_cap:.2f} 元")
    print(f"💵 Available Cash: {portfolio['financials']['cash']:.2f} 元")
    print("-" * 50)
    print(f"{'Symbol':<10} | {'Weight':<8} | {'Target Value':<12} | {'Current Value':<12} | {'Action'}")
    
    for s in symbols:
        weight = target_weights.get(s, 0.0)
        target_val = weight * total_cap
        price = prices.get(s, {}).get('close', 0)
        curr_val = current_shares.get(s, 0) * price
        
        diff_val = target_val - curr_val
        action = "WAIT"
        if diff_val > (price * 100):
            action = f"BUY {int(diff_val//(price*100))} lots"
        elif diff_val < -(price * 100):
            action = f"SELL {int(abs(diff_val)//(price*100))} lots"
            
        print(f"{s:<10} | {weight:<8.2f} | {target_val:<12.2f} | {curr_val:<12.2f} | {action}")

if __name__ == "__main__":
    debug_weights()
