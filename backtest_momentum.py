import pandas as pd
import numpy as np
import os
import sys
import json
import math

# 设置编码
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_DIR = os.path.join(BASE_DIR, 'history_data')
UNIVERSE_FILE = os.path.join(BASE_DIR, 'skynet_universe.json')

def run_backtest_momentum(min_trade_val=500.0, deadband=0.01):
    # 1. 加载资产清单
    with open(UNIVERSE_FILE, 'r', encoding='utf-8') as f:
        universe = json.load(f)
    symbols = list(universe['assets'].keys())
    
    # 2. 加载历史数据
    data_map = {}
    for sym in symbols:
        csv_path = os.path.join(HISTORY_DIR, f"{sym}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            data_map[sym] = df['close']
    
    price_df = pd.DataFrame(data_map).ffill().dropna()
    test_dates = price_df.index[-250:]
    
    # 3. 初始化账户
    initial_capital = 10000.0
    cash = initial_capital
    portfolio = {sym: 0.0 for sym in symbols}
    
    # 记录数据
    trade_count = 0
    total_fees = 0.0
    
    # 配置参数
    FEE_MIN = 5.0
    FEE_RATE = 0.00025
    
    # 动态调仓逻辑：每20天更新一次“盾+弩”权重
    for i, date in enumerate(test_dates):
        current_prices = price_df.loc[date]
        total_value = cash + sum([portfolio[s] * current_prices[s] for s in symbols])
        
        # 1. 计算当前最优权重
        # 简单逻辑：每20天重算一次 Top 3
        if i % 20 == 0:
            momentum_data = []
            for sym in symbols:
                if sym == "518880": continue # 排除黄金单独处理
                # 计算近20日收益
                series = price_df.loc[:date, sym]
                if len(series) > 20:
                    ret = (series.iloc[-1] - series.iloc[-21]) / series.iloc[-21]
                    momentum_data.append({"sym": sym, "ret": ret})
            
            momentum_data.sort(key=lambda x: x['ret'], reverse=True)
            top_2 = momentum_data[:2] # 取前两名
            
            # 目标权重分配
            target_weights = {
                "518880": 0.30, # 黄金 30%
                "510880": 0.20, # 红利 20%
                top_2[0]['sym']: 0.25,
                top_2[1]['sym']: 0.25
            }
            # 补全其他
            for s in symbols:
                if s not in target_weights: target_weights[s] = 0.0
        
        # 2. 调仓逻辑
        for s, target_w in target_weights.items():
            curr_w = (portfolio[s] * current_prices[s]) / total_value
            diff_w = abs(target_w - curr_w)
            trade_val = diff_w * total_value
            
            # 较敏锐的门槛：偏离 > 1% 且 交易额 > 500 元
            if diff_w >= deadband and trade_val >= min_trade_val:
                target_money = target_w * total_value
                old_shares = portfolio[s]
                new_shares = (target_money // (current_prices[s] * 100)) * 100
                delta_shares = new_shares - old_shares
                
                if delta_shares != 0:
                    trade_cost = abs(delta_shares * current_prices[s])
                    portfolio[s] = new_shares
                    cash -= (delta_shares * current_prices[s])
                    fee = max(FEE_MIN, trade_cost * FEE_RATE)
                    cash -= fee
                    total_fees += fee
                    trade_count += 1
    
    final_value = cash + sum([portfolio[s] * price_df.iloc[-1][s] for s in symbols])
    return final_value, trade_count, total_fees

if __name__ == "__main__":
    val, trades, fees = run_backtest_momentum(500.0, 0.01)
    print(f"--- 🚀 SkyNet '动态灵敏' 回测 (500元门槛 / 1%死区 / 每月重组) ---")
    print(f"最终净值: {val:.2f} 元")
    print(f"总交易次数: {trades} 次")
    print(f"总手续费: {fees:.2f} 元")
    print(f"平均每月交易: {trades/12:.1f} 次")
