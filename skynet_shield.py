
import pandas as pd
import akshare as ak
import json
import os
import datetime
import requests
import time

# 配置路径
PORTFOLIO_FILE = 'skynet_portfolio.json'
WEBHOOK_URL = "https://oapi.dingtalk.com/robot/send?access_token=d954c079fa4b880330ae13fde3d474044f673763670d47a127d775427e0c4462"

def send_dingtalk_alert(text):
    payload = {"msgtype": "markdown", "markdown": {"title": "🚨 紧急熔断预警", "text": text}}
    try:
        requests.post(WEBHOOK_URL, json=payload, timeout=10)
    except Exception as e:
        print(f"Failed to send DingTalk alert: {e}")

def morning_circuit_breaker():
    print("[SkyNet Shield] Starting Morning Circuit Breaker Monitor...")
    
    if not os.path.exists(PORTFOLIO_FILE):
        print(f"Portfolio file not found: {PORTFOLIO_FILE}")
        return

    try:
        with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
            portfolio = json.load(f)
    except Exception as e:
        print(f"Failed to load portfolio: {e}")
        return
    
    symbols_dict = portfolio.get('positions_in_shares', {})
    if not symbols_dict:
        print("No active positions.")
        return

    symbols = list(symbols_dict.keys())
    print(f"Monitoring {len(symbols)} symbols...")
    
    alert_list = []
    threshold = -0.03  # Trigger at -3%
    
    results = []
    for sym in symbols:
        try:
            # Use Sina Single Point Interface (Fast and Reliable)
            prefix = 'sh' if sym.startswith('5') or sym.startswith('6') or sym.startswith('11') else 'sz'
            url = f"http://hq.sinajs.cn/list={prefix}{sym}"
            headers = {'Referer': 'http://finance.sina.com.cn'}
            r = requests.get(url, headers=headers, timeout=5)
            
            # Extract data from Sina response
            # Format: var hq_str_sh510300="300ETF,3.882,3.894,3.882,...";
            content = r.text.split('"')[1]
            if not content:
                print(f"No data for {sym}")
                continue
            data = content.split(',')
            
            name = data[0]
            curr_price = float(data[3])
            prev_close = float(data[2])
            
            if prev_close == 0: continue
            pct_chg = (curr_price - prev_close) / prev_close

            results.append({
                'symbol': sym,
                'name': name,
                'pct_chg': pct_chg
            })
            print(f" - {sym} ({name}): {pct_chg*100:.2f}%")
        except Exception as ex:
            print(f"Error fetching {sym}: {ex}")

    if not results:
        print("No real-time data retrieved.")
        return

    for item in results:
        if item['pct_chg'] <= threshold:
            alert_list.append(f"- 🔴 **{item['name']}** ({item['symbol']}) 实时跌幅 **{item['pct_chg']*100:.2f}%**")

    if alert_list:
        msg = f"### 🚨 基地紧急熔断预警 [{datetime.datetime.now().strftime('%H:%M')}]\n\n"
        msg += "检测到早盘波动异常，以下资产已触及熔断阈值 (-3%)：\n\n"
        msg += "\n".join(alert_list)
        msg += "\n\n---\n💡 **建议**：当前市场异常，请关注仓位安全。紧急熔断程序已就绪。"
        print("\n[ALERT TRIGGERED]")
        send_dingtalk_alert(msg)
    else:
        print("\nMarket conditions normal. No circuit breaker triggered.")

if __name__ == "__main__":
    morning_circuit_breaker()
