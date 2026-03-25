import json
import os
import requests
from datetime import datetime

# 配置：阈值设定为 -2.0% (跌幅超过 2% 则报警)
LOSS_THRESHOLD = -2.0

# 路径配置
PORTFOLIO_PATH = "skynet_core/skynet_portfolio.json"
PRICES_PATH = "skynet_core/skynet_daily_prices.json"
DINGTALK_WEBHOOK = "https://oapi.dingtalk.com/robot/send?access_token=d954c079fa4b880330ae13fde3d474044f673763670d47a127d775427e0c4462"

def send_dingtalk(message):
    payload = {
        "msgtype": "text",
        "text": {"content": f"🚨 [SkyNet 盘前预警] 🐥\n{message}"}
    }
    try:
        requests.post(DINGTALK_WEBHOOK, json=payload, timeout=5)
    except Exception as e:
        print(f"DingTalk error: {e}")

def get_realtime_price(code):
    """
    极简实时价格获取 (模拟，实际可调用腾讯/新浪接口)
    这里为了 09:15 能够跑通，建议使用腾讯 API: http://qt.gtimg.cn/q=s_shxxxxxx
    """
    prefix = "sh" if code.startswith("51") or code.startswith("60") else "sz"
    url = f"http://qt.gtimg.cn/q=s_{prefix}{code}"
    try:
        resp = requests.get(url, timeout=5)
        # 解析格式: v_s_sh510300="1~沪深300ETF~510300~3.876~-0.004~-0.10~721094~279471~~";
        data = resp.text.split("~")
        if len(data) > 3:
            return {
                "price": float(data[3]),
                "change_percent": float(data[5])
            }
    except:
        return None
    return None

def run_sentry():
    if not os.path.exists(PORTFOLIO_PATH):
        print("Portfolio not found.")
        return

    with open(PORTFOLIO_PATH, 'r', encoding='utf-8') as f:
        portfolio = json.load(f)
    
    with open(PRICES_PATH, 'r', encoding='utf-8') as f:
        prev_prices = json.load(f)
    
    positions = portfolio.get("positions_in_shares", {})
    active_positions = {k: v for k, v in positions.items() if v > 0}
    
    if not active_positions:
        print("No active positions to monitor.")
        return

    alerts = []
    print(f"Checking {len(active_positions)} positions at {datetime.now()}...")
    
    for code in active_positions:
        data = get_realtime_price(code)
        if data:
            current_price = data['price']
            # 使用本地记录的昨日收盘价计算真实涨跌幅
            prev_close = prev_prices.get(code, {}).get('close')
            if prev_close:
                real_pct = round((current_price - prev_close) / prev_close * 100, 2)
                print(f"[{code}] Current: {current_price}, PrevClose: {prev_close}, RealChange: {real_pct}%")
                
                if real_pct <= LOSS_THRESHOLD:
                    alerts.append(f"⚠️ {code} 跌幅告急: {real_pct}% (当前价: {current_price})")
            else:
                print(f"No prev close record for {code}, skipping calculation.")
        else:
            print(f"Could not fetch data for {code}")

    if alerts:
        msg = "\n".join(alerts) + "\n\n建议动作：考虑减仓或全额撤退以规避风险。"
        send_dingtalk(msg)
        print("Alerts sent to DingTalk.")
    else:
        # 修复：不再发送 active_positions 对象，而是发送具体的涨跌幅数据
        status_lines = []
        for code in active_positions:
            data = get_realtime_price(code)
            prev_close = prev_prices.get(code, {}).get('close')
            if data and prev_close:
                pct = round((data['price'] - prev_close) / prev_close * 100, 2)
                status_lines.append(f"- [{code}]: {pct}% (当前: {data['price']})")
        
        msg = "✅ 持仓巡逻完毕，当前行情健康，未发现大幅低开信号。\n" + "\n".join(status_lines)
        send_dingtalk(msg)
        print("Healthy status sent to DingTalk.")

if __name__ == "__main__":
    run_sentry()
