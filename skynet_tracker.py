
import json
import datetime
import os

PORTFOLIO_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'skynet_portfolio.json')
PRICE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'skynet_daily_prices.json')
TRACKER_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'skynet_performance_log.json')

def update_tracker():
    if not os.path.exists(PORTFOLIO_FILE) or not os.path.exists(PRICE_FILE):
        print("Missing portfolio or price data.")
        return

    with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
        portfolio = json.load(f)
    with open(PRICE_FILE, 'r', encoding='utf-8') as f:
        prices = json.load(f)

    # Calculate current value
    total_holdings_value = 0
    details = {}
    
    for sym, shares in portfolio['positions_in_shares'].items():
        if sym in prices:
            price = prices[sym]['close']
            name = prices[sym]['name']
            val = shares * price
            total_holdings_value += val
            details[sym] = {
                "name": name,
                "shares": shares,
                "price": price,
                "value": round(val, 2)
            }

    total_value = total_holdings_value + portfolio['financials']['cash']
    today_str = datetime.datetime.now().strftime('%Y-%m-%d')

    # Load history
    history = []
    if os.path.exists(TRACKER_LOG):
        with open(TRACKER_LOG, 'r', encoding='utf-8') as f:
            history = json.load(f)

    # Record entry
    entry = {
        "date": today_str,
        "total_value": round(total_value, 2),
        "cash": portfolio['financials']['cash'],
        "holdings_value": round(total_holdings_value, 2),
        "details": details
    }
    
    # Update or append
    if history and history[-1]['date'] == today_str:
        history[-1] = entry
    else:
        history.append(entry)

    with open(TRACKER_LOG, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

    summary_lines = []
    summary_lines.append(f"### 📈 今日收盘表现 ({today_str})")
    if len(history) > 1:
        prev_value = history[-2]['total_value']
        pct_change = (total_value - prev_value) / prev_value * 100
        summary_lines.append(f"**较前日涨跌**: {pct_change:+.2f}%")
        summary_lines.append(f"**今日盈亏额**: {total_value - prev_value:+.2f} 元")
    else:
        summary_lines.append(f"**较前日涨跌**: N/A (首日记录)")
    
    summary_lines.append("\n**[当前持仓快照]**:")
    for sym, d in details.items():
        if d['shares'] > 0:
            summary_lines.append(f"- {d['name']} ({sym}): {d['shares']}股, 市值 {d['value']}元")
    
    summary_str = "\n".join(summary_lines)
    
    # Keep print for agent log
    print("REPORT_START")
    print(summary_str)
    print("REPORT_END")
    
    return summary_str

if __name__ == "__main__":
    update_tracker()
