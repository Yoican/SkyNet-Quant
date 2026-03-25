
import json
import datetime
import os
import sys

# 强制使用 UTF-8 编码输出，防止 Windows 环境下的乱码
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ---------------------------------------------------------
# 配置路径 (基于 skynet_core 目录)
# ---------------------------------------------------------
PORTFOLIO_FILE = 'skynet_portfolio.json'
PRICE_FILE = 'skynet_daily_prices.json'
PERFORMANCE_LOG = 'skynet_performance_history.json'

def update_portfolio_and_stats():
    print("🐥 [SkyNet Tracker] 正在启动资产结算系统...")
    
    if not os.path.exists(PORTFOLIO_FILE) or not os.path.exists(PRICE_FILE):
        print("❌ 错误：未找到持仓文件或价格文件。请先运行价格更新脚本。")
        return

    # 1. 加载数据
    with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
        portfolio = json.load(f)
    with open(PRICE_FILE, 'r', encoding='utf-8') as f:
        prices = json.load(f)

    # 2. 计算当前市值
    current_holdings_value = 0.0
    details = []
    
    # 获取当前日期
    today_str = datetime.datetime.now().strftime('%Y-%m-%d')
    
    print(f"📊 结算日期: {today_str}")
    print("-" * 40)

    for sym, shares in portfolio['positions_in_shares'].items():
        if shares <= 0: continue
        
        if sym in prices:
            price = prices[sym]['close']
            name = prices[sym]['name']
            market_value = shares * price
            current_holdings_value += market_value
            details.append({
                "symbol": sym,
                "name": name,
                "shares": shares,
                "price": price,
                "market_value": round(market_value, 2)
            })
        else:
            print(f"⚠️ 警告：缺少资产 {sym} 的最新价格信息")

    total_assets = current_holdings_value + portfolio['financials']['cash']
    
    # 3. 更新内存中的 portfolio 文件
    portfolio['financials']['holdings_value'] = round(current_holdings_value, 2)
    portfolio['financials']['total_value'] = round(total_assets, 2)
    portfolio['last_update'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(PORTFOLIO_FILE, 'w', encoding='utf-8') as f:
        json.dump(portfolio, f, indent=4, ensure_ascii=False)
    
    # 4. 记录历史走势
    history = []
    if os.path.exists(PERFORMANCE_LOG):
        try:
            with open(PERFORMANCE_LOG, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except:
            history = []

    # 如果当天已记录，则更新；否则追加
    new_entry = {
        "date": today_str,
        "total_value": round(total_assets, 2),
        "cash": portfolio['financials']['cash'],
        "holdings_value": round(current_holdings_value, 2)
    }
    
    if history and history[-1]['date'] == today_str:
        history[-1] = new_entry
    else:
        history.append(new_entry)
        
    with open(PERFORMANCE_LOG, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

    # 5. 生成可视化报表 (控制台输出)
    print(f"{'资产名称':<10} | {'持仓(股)':<8} | {'现价':<8} | {'市值(元)':<10}")
    print("-" * 40)
    for d in details:
        print(f"{d['name']:<10} | {d['shares']:<10} | {d['price']:<10.3f} | {d['market_value']:<10.2f}")
    
    print("-" * 40)
    print(f"💵 剩余现金: {portfolio['financials']['cash']:.2f} 元")
    print(f"💰 总资产估值: {total_assets:.2f} 元")
    
    if len(history) > 1:
        prev_val = history[-2]['total_value']
        pct_change = (total_assets - prev_val) / prev_val * 100
        diff = total_assets - prev_val
        print(f"📈 今日涨跌: {diff:+.2f} 元 ({pct_change:+.2f}%)")
    
    print("\n✅ 持仓信息已同步，历史走势已记录。")

if __name__ == "__main__":
    update_portfolio_and_stats()
