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

def run_backtest():
    # 1. 加载资产清单
    with open(UNIVERSE_FILE, 'r', encoding='utf-8') as f:
        universe = json.load(f)
    symbols = list(universe['assets'].keys())
    
    # 2. 加载历史数据并对齐日期
    data_map = {}
    for sym in symbols:
        csv_path = os.path.join(HISTORY_DIR, f"{sym}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            data_map[sym] = df['close']
    
    price_df = pd.DataFrame(data_map).ffill().dropna()
    # 选取最近 250 个交易日（约一年）进行压力测试
    test_dates = price_df.index[-250:]
    
    # 3. 初始化账户
    initial_capital = 10000.0
    cash = initial_capital
    portfolio = {sym: 0.0 for sym in symbols}
    total_value = initial_capital
    
    # 记录数据
    history_values = []
    trade_count = 0
    total_fees = 0.0
    last_executed_weights = {sym: 0.0 for sym in symbols}
    
    # 配置参数
    FEE_MIN = 5.0 # 最低 5 元
    FEE_RATE = 0.00025 # 万 2.5
    DEADBAND = 0.03 # 3% 死区
    MIN_TRADE_VAL = 2500.0 # 2500 元交易门槛
    
    # 预定义的“盾+弩”目标权重
    # 黄金(30%), 红利(20%), 东南亚科技(25%), 纳指100(25%)
    target_weights = {
        "518880": 0.30, "510880": 0.20, "513730": 0.25, "513100": 0.25
    }
    # 填充其他标的为 0
    for s in symbols:
        if s not in target_weights: target_weights[s] = 0.0

    print(f"开始回测: {test_dates[0].date()} 至 {test_dates[-1].date()}")
    
    for i, date in enumerate(test_dates):
        # 更新每日市值
        current_prices = price_df.loc[date]
        holdings_value = sum([portfolio[s] * current_prices[s] for s in symbols])
        total_value = cash + holdings_value
        
        # 调仓决策逻辑 (模拟 Deadband + Min Trade Value)
        # 假设我们每天计算，但只有满足条件才执行
        if i == 0:
            # 第一天强制建仓 (初始重组)
            for s, w in target_weights.items():
                if w > 0:
                    target_money = w * total_value
                    shares = (target_money // (current_prices[s] * 100)) * 100
                    cost = shares * current_prices[s]
                    portfolio[s] = shares
                    cash -= cost
                    # 计费
                    fee = max(FEE_MIN, cost * FEE_RATE)
                    cash -= fee
                    total_fees += fee
                    trade_count += 1
            last_executed_weights = {s: (portfolio[s]*current_prices[s]/total_value) for s in symbols}
        else:
            # 检查是否需要调仓
            for s, target_w in target_weights.items():
                curr_w = (portfolio[s] * current_prices[s]) / total_value
                diff_w = abs(target_w - curr_w)
                
                # 触发止损或超过 2500 元门槛且满足死区
                trade_val = abs(target_w - curr_w) * total_value
                
                # 逻辑：止损权重归零 OR (偏离>3% AND 交易额>2500)
                if (target_w == 0 and portfolio[s] > 0) or (diff_w >= DEADBAND and trade_val >= MIN_TRADE_VAL):
                    # 执行调仓
                    target_money = target_w * total_value
                    old_shares = portfolio[s]
                    new_shares = (target_money // (current_prices[s] * 100)) * 100
                    delta_shares = new_shares - old_shares
                    
                    if delta_shares != 0:
                        trade_cost = abs(delta_shares * current_prices[s])
                        portfolio[s] = new_shares
                        cash -= (delta_shares * current_prices[s])
                        # 计费
                        fee = max(FEE_MIN, trade_cost * FEE_RATE)
                        cash -= fee
                        total_fees += fee
                        trade_count += 1
        
        history_values.append(total_value)

    # 4. 统计结果
    final_value = history_values[-1]
    total_return = (final_value - initial_capital) / initial_capital
    
    print("\n--- 📊 '盾+弩' 绝不消失计划 回测报告 ---")
    print(f"回测时长: {len(test_dates)} 交易日 (~1年)")
    print(f"初始本金: {initial_capital:.2f} 元")
    print(f"最终净值: {final_value:.2f} 元")
    print(f"累计收益率: {total_return:.2%}")
    print(f"累计调仓次数: {trade_count} 次")
    print(f"累计手续费损耗: {total_fees:.2f} 元 (占本金 {total_fees/initial_capital:.2%})")
    print(f"平均每月交易: {trade_count/12:.1f} 次")

if __name__ == "__main__":
    run_backtest()
