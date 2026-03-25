import sys
import os

# 将根目录添加到 sys.path 以便导入 v11_modules
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import pandas as pd
import numpy as np
import xgboost as xgb
import io
import warnings
import json
import datetime
import requests
import math
from sklearn.covariance import LedoitWolf
import akshare as ak

# 导入 V11.0 模块
from v11_modules.deadband import PositionDeadband
from v11_modules.mcr import MarginalContributionRiskConstraint
from v11_modules.hmm_regime import HMMMarketRegimeDetector
from v11_modules.xgboost_online import XGBoostOnlineLearner

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 配置与路径
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PORTFOLIO_FILE = os.path.join(BASE_DIR, 'skynet_portfolio.json')
PRICE_FILE = os.path.join(BASE_DIR, 'skynet_daily_prices.json')
UNIVERSE_FILE = os.path.join(BASE_DIR, 'skynet_universe.json')
STATE_FILE = os.path.join(BASE_DIR, 'skynet_state.json')
HISTORY_DIR = os.path.join(BASE_DIR, 'history_data')
WEBHOOK_URL = "https://oapi.dingtalk.com/robot/send?access_token=d954c079fa4b880330ae13fde3d474044f673763670d47a127d775427e0c4462"

def send_dingtalk_markdown(title, text):
    headers = {'Content-Type': 'application/json'}
    payload = {"msgtype": "markdown", "markdown": {"title": title, "text": text}}
    try:
        requests.post(WEBHOOK_URL, headers=headers, data=json.dumps(payload), timeout=5)
    except:
        pass

# ---------------------------------------------------------
# 1. 持仓与价格感知模块
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 2. 增强型指令生成逻辑
# ---------------------------------------------------------
def calculate_trade_instructions(target_weights, portfolio, prices):
    total_capital = portfolio['financials']['total_value']
    current_cash = portfolio['financials']['cash']
    current_positions = portfolio['positions_in_shares']
    
    orders = []
    new_shares_map = {}
    
    # 摩擦力控制参数 (已校准：针对 9000 元总盘子优化)
    MIN_TRADE_VALUE = 800.0  # 恢复 800 元高门槛，严格控制操作频率
    
    # 全额建仓逻辑判定：如果现金大于 500 元，视为正在进行大额补齐
    is_initial_build = (current_cash > 500) 
    
    for sym_str, weight in target_weights.items():
        if sym_str not in prices or "close" not in prices[sym_str]:
            continue
        
        price = prices[sym_str]['close']
        target_money = weight * total_capital
        target_shares = math.floor(target_money / (price * 100)) * 100
        new_shares_map[sym_str] = target_shares
        
        current_shares = current_positions.get(sym_str, 0)
        delta_shares = target_shares - current_shares
        trade_value = abs(delta_shares * price)
        
        # 核心逻辑：动态门槛逻辑
        is_initial_build = (current_cash > 500) # 3-19 全额建仓期
        is_stop_loss = (weight == 0) # 止损清仓，哪怕 5 元也要撤
        
        if delta_shares != 0:
            if trade_value >= MIN_TRADE_VALUE or is_initial_build or is_stop_loss:
                orders.append({
                    "symbol": sym_str,
                    "name": prices[sym_str]['name'],
                    "delta": delta_shares,
                    "price": price,
                    "value": delta_shares * price
                })
            else:
                # 交易额太小，不足以覆盖 5 元，维持现状
                new_shares_map[sym_str] = current_shares
            
    orders.sort(key=lambda x: x['value']) 
    
    MIN_CASH_RESERVE = 100.0 
    executable_instructions = []
    temp_cash = current_cash - MIN_CASH_RESERVE
    
    for o in orders:
        if o['delta'] > 0: # 买入
            cost = o['value']
            if temp_cash >= cost:
                temp_cash -= cost
                executable_instructions.append(f"- **{o['name']} ({o['symbol']})**: 🟢 买入 **{abs(o['delta'])}** 份 ({abs(o['delta'])//100} 手)")
            else:
                possible_lots = math.floor(temp_cash / (o['price'] * 100))
                if possible_lots > 0:
                    real_delta = possible_lots * 100
                    temp_cash -= (real_delta * o['price'])
                    executable_instructions.append(f"- **{o['name']} ({o['symbol']})**: ⚠️ 限额买入 **{real_delta}** 份 ({possible_lots} 手)")
        else: # 卖出
            temp_cash += abs(o['value'])
            executable_instructions.append(f"- **{o['name']} ({o['symbol']})**: 🔴 卖出 **{abs(o['delta'])}** 份 ({abs(o['delta'])//100} 手)")
            
    return executable_instructions, new_shares_map

# ---------------------------------------------------------
# 数据读取与特征加工
# ---------------------------------------------------------
def fetch_and_prep_daily_data(symbols):
    all_data = []
    if not os.path.exists(HISTORY_DIR): os.makedirs(HISTORY_DIR)

    for sym in symbols:
        csv_path = os.path.join(HISTORY_DIR, f"{sym}.csv")
        try:
            df = pd.read_csv(csv_path)
            df['date'] = pd.to_datetime(df['date'])
            df['symbol'] = int(sym)
            df['pct_chg_1d'] = df['close'].pct_change()
            df['pct_chg_5d'] = df['close'].pct_change(periods=5)
            df['ma60'] = df['close'].rolling(60).mean()
            df['volatility_10d'] = df['pct_chg_1d'].rolling(10).std()
            all_data.append(df.dropna().reset_index(drop=True))
        except Exception as e:
            print(f"⚠️ {sym} CSV loading failed: {e}")
            
    return pd.concat(all_data, axis=0) if all_data else pd.DataFrame()

# ---------------------------------------------------------
# 主流程
# ---------------------------------------------------------
def run_skynet_live():
    # Fix encoding for Windows console
    print("🚀 [V11.1 Pro] 天网实盘决策系统启动...")
    
    with open(UNIVERSE_FILE, 'r', encoding='utf-8') as f:
        universe = json.load(f)
    symbols = list(universe['assets'].keys())

    # 1. 强制数据获取
    print("🔍 [SkyNet Data Guard] Ensuring full OHLCV history via Rapid Sync...")
    try:
        import subprocess
        sync_script = os.path.join(BASE_DIR, "rapid_sync.py")
        result = subprocess.run(["python", sync_script], capture_output=True, text=True, check=True)
        print(result.stdout)
        
        # 2. 6-field 校验
        print("🛡️ [Data Guard] Performing 6-field OHLCV Integrity Check...")
        missing_assets = []
        for sym in symbols:
            csv_path = os.path.join(HISTORY_DIR, f"{sym}.csv")
            if not os.path.exists(csv_path):
                missing_assets.append(f"{sym}(Missing CSV)")
                continue
            check_df = pd.read_csv(csv_path)
            if check_df.empty:
                missing_assets.append(f"{sym}(Empty)")
                continue
            last_row = check_df.iloc[-1]
            for field in ['date', 'open', 'close', 'high', 'low', 'vol']:
                if field not in check_df.columns or pd.isna(last_row[field]):
                    missing_assets.append(f"{sym}(Missing {field})")
                    break
        
        if missing_assets:
            err_msg = f"❌ [DATA CRITICAL] 决策终止！数据不全: {', '.join(missing_assets)}"
            print(err_msg)
            send_dingtalk_markdown("⚠️ 天网发车失败", err_msg)
            return

    except Exception as e:
        err_msg = f"❌ [SYSTEM ERROR] 数据同步异常: {e}"
        print(err_msg)
        send_dingtalk_markdown("🚨 系统异常", err_msg)
        return

    portfolio, current_prices = load_live_context()
    
    try:
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            state = json.load(f)
    except:
        state = {"smoothed_weights": {}}
    
    last_smoothed_weights = {k: v for k, v in state.get('smoothed_weights', {}).items()}
    
    # 3. 计算特征与权重
    # --- 收益增强型初始持仓逻辑覆盖 ---
    import sys
    sys.path.append(BASE_DIR)
    try:
        from best_setup_analysis import analyze_best_setup
        target_weights, top_assets = analyze_best_setup()
        print("💡 [Alpha Override] 收益增强型配置已激活")
    except Exception as e:
        print(f"⚠️ 增强方案加载失败: {e}")
        target_weights = {"518880": 0.30}
        rem = 0.7 / (len(symbols)-1)
        for s in symbols:
            if s != "518880": target_weights[s] = rem

    alpha = 0.5 # 策略归位：恢复稳健的平滑过滤模式 (Safety Mode)
    smoothed_weights = {}
    for s in symbols:
        curr = target_weights.get(s, 0.0)
        # 如果是清仓标的，curr 为 0
        last = last_smoothed_weights.get(s, curr)
        smoothed_weights[s] = alpha * curr + (1 - alpha) * last

    # 4. 指令生成与保存
    # 引入死区过滤 (Deadband) 降低摩擦成本
    deadband = PositionDeadband(threshold=0.03) # 3.0% 死区，恢复默认校准值
    
    # 摩擦力控制参数 (已校准：针对 9000 元总盘子优化)
    MIN_TRADE_VALUE = 800.0  
    # 3. 死区阈值: 3.0% (偏离 3% 且 交易额 > 800 才动)
    # ---------------------------------------------------------
    if state.get('smoothed_weights'):
        deadband.current_weights = state['smoothed_weights']
    
    final_execution_weights = deadband.apply(smoothed_weights)
    
    instructions, new_shares = calculate_trade_instructions(final_execution_weights, portfolio, current_prices)

    portfolio['positions_in_shares'] = new_shares
    portfolio['last_update'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(PORTFOLIO_FILE, 'w', encoding='utf-8') as f:
        json.dump(portfolio, f, indent=4, ensure_ascii=False)

    # 保存今日实时价格供 Tracker 使用
    price_export = {}
    for sym, p in current_prices.items():
        price_export[sym] = {"name": p['name'], "close": p['close'], "date": datetime.datetime.now().strftime('%Y-%m-%d')}
    with open(PRICE_FILE, 'w', encoding='utf-8') as f:
        json.dump(price_export, f, indent=4, ensure_ascii=False)

    # 尝试更新 Tracker
    tracker_summary = ""
    try:
        from skynet_tracker import update_tracker
        tracker_summary = update_tracker() or ""
    except Exception as e:
        print(f"Tracker error: {e}")

    report = f"### ⏱️ 实盘决策 (已校准): {datetime.datetime.now().strftime('%Y-%m-%d')}\n\n"
    report += f"**💰 总资产**: {portfolio['financials']['total_value']:.2f} 元 | **💵 现金**: {portfolio['financials']['cash']:.2f} 元\n"
    report += f"**🛡️ 摩擦优化**: 已启用 3% 调仓死区 (Deadband)\n"
    
    if tracker_summary:
        report += f"\n{tracker_summary}\n"

    if instructions:
        report += "\n### 🚨 建议操作指令\n" + "\n".join(instructions)
    else:
        report += "\n### 💤 暂无操作 (变动未达 3% 阈值，已过滤摩擦损耗)"

    print(report)
    send_dingtalk_markdown("天网 V11.1 资产对齐报表", report)
    
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump({"smoothed_weights": final_execution_weights, "last_update": portfolio['last_update']}, f, indent=4)

if __name__ == "__main__":
    run_skynet_live()
