
import json
import math

PORTFOLIO_FILE = 'skynet_portfolio.json'
PRICE_FILE = 'skynet_daily_prices.json'

def invest_remaining_cash():
    with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
        portfolio = json.load(f)
    with open(PRICE_FILE, 'r', encoding='utf-8') as f:
        prices = json.load(f)
    
    cash = portfolio['financials']['cash']
    if cash < 100:
        print("现金不足 100 元，暂无法进行调仓。")
        return

    # 优先补齐 V12 里的避险资产：国开债 ETF (159649) 和 短债 ETF (511010)
    # 或是根据当前 V12 逻辑分配到 纳指/标普
    targets = [
        {"sym": "513100", "name": "纳指100", "weight": 0.4},
        {"sym": "513500", "name": "标普500", "weight": 0.3},
        {"sym": "513520", "name": "日经ETF", "weight": 0.3}
    ]
    
    print(f"### 💰 现金全额投产建议 (针对余额: {cash:.2f} 元)")
    instructions = []
    
    for t in targets:
        sym = t['sym']
        if sym in prices:
            price = prices[sym]['close']
            # 计算能买多少手
            target_money = cash * t['weight']
            lots = math.floor(target_money / (price * 100))
            if lots > 0:
                shares = lots * 100
                cost = shares * price
                instructions.append(f"- **{t['name']} ({sym})**: 🟢 建议买入 **{shares}** 份 ({lots} 手) [预计耗资 {cost:.2f} 元]")

    if not instructions:
        # 如果权重分配太细买不起，尝试全买标普
        sym = "513500"
        price = prices[sym]['close']
        lots = math.floor(cash / (price * 100))
        if lots > 0:
            instructions.append(f"- **标普500 ({sym})**: 🟢 建议全额买入 **{lots*100}** 份 ({lots} 手)")
    
    for ins in instructions:
        print(ins)

if __name__ == "__main__":
    invest_remaining_cash()
