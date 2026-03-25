
import pandas as pd
import numpy as np
import sys
import io
import warnings
import json
import datetime
import requests
import math
import akshare as ak
import os

# 导入情绪插件 (动态路径)
sys.path.append(os.path.abspath('../skills/market-sentiment/scripts'))
try:
    import market_sentiment
except ImportError:
    market_sentiment = None

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ---------------------------------------------------------
# 配置与路径
# ---------------------------------------------------------
PORTFOLIO_FILE = 'skynet_portfolio.json'
PRICE_FILE = 'skynet_daily_prices.json'
UNIVERSE_FILE = 'skynet_universe.json'
STATE_FILE = 'skynet_state.json'
WEBHOOK_URL = "https://oapi.dingtalk.com/robot/send?access_token=d954c079fa4b880330ae13fde3d474044f673763670d47a127d775427e0c4462"

def send_dingtalk_markdown(title, text):
    headers = {'Content-Type': 'application/json'}
    payload = {"msgtype": "markdown", "markdown": {"title": title, "text": text}}
    try:
        requests.post(WEBHOOK_URL, headers=headers, data=json.dumps(payload), timeout=5)
    except:
        pass

def load_live_context():
    try:
        with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
            portfolio = json.load(f)
    except:
        portfolio = {"financials": {"total_value": 10000, "cash": 10000}, "positions_in_shares": {}}
    try:
        with open(PRICE_FILE, 'r', encoding='utf-8') as f:
            prices = json.load(f)
    except:
        prices = {}
    return portfolio, prices

def get_market_sentiment_multiplier():
    """从插件获取情绪乘数：恐惧时增加防守，贪婪时注意止盈"""
    if not market_sentiment:
        return 1.0, "插件未激活"
    
    try:
        fgi = market_sentiment.get_fear_greed_index()
        score = fgi['value']
        label = fgi['label']
        
        # 逻辑：
        # Score < 30 (极度恐惧) -> 避险乘数 1.5 (增加债基比例)
        # Score > 70 (极度贪婪) -> 避险乘数 1.3 (落袋为安)
        # 40-60 (中性) -> 1.0
        if score < 30: multiplier = 1.5
        elif score < 45: multiplier = 1.2
        elif score > 75: multiplier = 1.4
        elif score > 60: multiplier = 1.1
        else: multiplier = 1.0
        
        return multiplier, f"{score} ({label})"
    except:
        return 1.0, "获取失败"

def calculate_trade_instructions(target_weights, portfolio, prices):
    total_capital = portfolio['financials']['total_value']
    current_cash = portfolio['financials']['cash']
    current_positions = portfolio['positions_in_shares']
    
    orders = []
    for sym_str, weight in target_weights.items():
        if sym_str not in prices:
            print(f"Warning: {sym_str} not in price database.")
            continue
        if 'close' not in prices[sym_str]:
            print(f"Warning: 'close' price missing for {sym_str}.")
            continue
        price = prices[sym_str]['close']
        target_shares = math.floor((weight * total_capital) / (price * 100)) * 100
        current_shares = current_positions.get(sym_str, 0)
        delta = target_shares - current_shares
        if delta != 0:
            orders.append({"symbol": sym_str, "name": prices[sym_str]['name'], "delta": delta, "price": price, "value": delta * price})

    orders.sort(key=lambda x: x['value'])
    
    executable = []
    temp_cash = current_cash
    for o in orders:
        if o['delta'] < 0:
            temp_cash += abs(o['value'])
            executable.append(f"- **{o['name']} ({o['symbol']})**: 🔴 卖出 **{abs(o['delta'])}** 份 ({abs(o['delta'])//100} 手)")
        else:
            if temp_cash >= o['value']:
                temp_cash -= o['value']
                executable.append(f"- **{o['name']} ({o['symbol']})**: 🟢 买入 **{abs(o['delta'])}** 份 ({abs(o['delta'])//100} 手)")
            else:
                lots = math.floor(temp_cash / (o['price'] * 100))
                if lots > 0:
                    temp_cash -= (lots * 100 * o['price'])
                    executable.append(f"- **{o['name']} ({o['symbol']})**: ⚠️ 限额买入 **{lots*100}** 份 ({lots} 手)")
    return executable

def run_v12_plus():
    print("🚀 [V12.1 Sentiment+ Macro] 天网情绪增强版启动...")
    portfolio, current_prices = load_live_context()
    
    # 获取情绪因子
    sentiment_mult, sentiment_desc = get_market_sentiment_multiplier()
    print(f"📊 当前市场情绪: {sentiment_desc} | 避险乘数: {sentiment_mult}")
    
    # 动态调整权重逻辑
    # 基础避险比例 (国开债 + 短债)
    base_safety_weight = 0.35 
    # 情绪调节后的避险比例 (上限 60%)
    dynamic_safety_weight = min(0.60, base_safety_weight * sentiment_mult)
    
    rem_weight = 1.0 - dynamic_safety_weight
    
    # 资产分配 (V12.1 情绪版)
    target_weights = {
        "159649": dynamic_safety_weight * 0.6, # 国开债
        "511010": dynamic_safety_weight * 0.4, # 短债
        "513100": rem_weight * 0.35, # 纳指
        "513500": rem_weight * 0.25, # 标普
        "518880": rem_weight * 0.15, # 黄金
        "510300": rem_weight * 0.15, # 沪深300
        "512760": rem_weight * 0.10  # 芯片
    }
    
    instructions = calculate_trade_instructions(target_weights, portfolio, current_prices)
    
    report = f"### ⏱️ SkyNet V12.1 情绪增强指令 [{datetime.datetime.now().strftime('%Y-%m-%d')}]\n\n"
    report += f"**🌡️ 市场情绪**: `{sentiment_desc}`\n"
    report += f"**🛡️ 防御等级**: `{'高' if sentiment_mult > 1.2 else '中'}` (避险权重: {dynamic_safety_weight*100:.1f}%)\n"
    report += f"**💰 实盘总资产**: {portfolio['financials']['total_value']} 元\n"
    report += "\n### 🚨 调仓动作 (已考虑全网情绪)\n" + ("\n".join(instructions) if instructions else "保持现状")
    report += "\n\n---\n*⚠️ 此为 V12.1 情绪增强版唯一官方指令。*"

    print(report)
    send_dingtalk_markdown("天网 V12.1 情绪增强版", report)

if __name__ == "__main__":
    # run_v12_plus() # 🐥 [V12.1 Lockdown] 暂时封印，严禁在 A 股实盘盲目使用
    print("⚠️ SkyNet V12.1 is currently in LAB mode. Please use V11.1 PRO for production.")
