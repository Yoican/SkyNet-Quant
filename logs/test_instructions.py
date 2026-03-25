import sys
import math

# 设置编码
sys.stdout.reconfigure(encoding='utf-8')

# 模拟 3-19 情况
total_value = 9793.6
current_cash = 2431.8
prices = {
    "518880": 10.606, "513730": 1.239, "513100": 1.789, "512760": 1.717,
    "513500": 2.284, "516770": 1.436, "513520": 1.941, "512880": 1.114,
    "515220": 1.251, "510300": 4.662, "512100": 3.240, "510880": 3.296
}

# 目标权重 (盾+弩方案)
target_weights = {
    "518880": 0.30, # 黄金
    "510880": 0.20, # 红利
    "513730": 0.25, # 东南亚
    "513100": 0.25  # 纳指
}

# 当前持仓
current_shares = {
    "513100": 900, "513500": 500, "516770": 400, "513520": 300,
    "512880": 500, "515220": 400, "510300": 300, "512100": 100,
    "512760": 200, "510880": 100, "518880": 0, "513730": 0
}

instructions = []
total_buy_value = 0
total_sell_value = 0

# 1. 卖出
for sym, shares in current_shares.items():
    if sym not in target_weights and shares > 0:
        val = shares * prices[sym]
        instructions.append(f"SELL {sym}: {shares} shares @ {prices[sym]} (Value: {val:.1f})")
        total_sell_value += val

# 2. 买入
for sym, weight in target_weights.items():
    target_money = weight * total_value
    current_money = current_shares.get(sym, 0) * prices[sym]
    diff_money = target_money - current_money
    
    if diff_money > 100:
        lots = math.floor(diff_money / (prices[sym] * 100))
        if lots > 0:
            shares_to_buy = lots * 100
            val = shares_to_buy * prices[sym]
            instructions.append(f"BUY {sym}: {shares_to_buy} shares @ {prices[sym]} (Value: {val:.1f})")
            total_buy_value += val

print("--- SKYNET INSTRUCTION PREVIEW ---")
for ins in instructions:
    print(ins)
print(f"Total Sell: {total_sell_value:.1f}")
print(f"Total Buy: {total_buy_value:.1f}")
print(f"Est Final Cash: {(current_cash + total_sell_value - total_buy_value):.1f}")
