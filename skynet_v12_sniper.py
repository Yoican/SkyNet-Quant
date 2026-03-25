import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(BASE_DIR, "v12_features")
PORTFOLIO_PATH = os.path.join(BASE_DIR, "skynet_portfolio.json")
UNIVERSE_PATH = os.path.join(BASE_DIR, "skynet_universe.json")
MODEL_PATH = os.path.join(BASE_DIR, "v12_st_transformer.pth")
META_PATH = os.path.join(BASE_DIR, "skynet_v12_meta.json")

# 引入模型架构
from v12_st_model import SpatialTemporalTransformer
from v12_st_pipeline import FEATURE_COLS, SEQ_LEN

# V12 机器寻优确定的实战参数
FEE_MIN = 5.0
BUY_Z_TH = 1.0                 # 普通开枪 Z-Score 阈值
HIGH_CONVICTION_Z_TH = 1.5     # 重拳出击 Z-Score 阈值
SELL_Z_TH = -0.5               # 趋势转弱卖出 Z-Score 阈值
MIN_TRADE_AMOUNT = 1500.0      # 最小允许交易金额 (元)
TRAIL_STOP_PCT = 0.04          # 高水位移动止盈阈值 (4%回撤卖出)

def load_portfolio():
    if os.path.exists(PORTFOLIO_PATH):
        with open(PORTFOLIO_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"cash": 10000.0, "positions_in_shares": {}}

def load_meta():
    if os.path.exists(META_PATH):
        with open(META_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_meta(meta):
    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

def load_universe():
    if os.path.exists(UNIVERSE_PATH):
        with open(UNIVERSE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f).get('assets', {})
    return {}

def get_latest_spatial_temporal_tensor():
    symbols = []
    raw_dfs = {}
    for filename in sorted(os.listdir(FEATURES_DIR)):
        if filename.endswith(".csv"):
            sym = filename.replace('.csv', '')
            df = pd.read_csv(os.path.join(FEATURES_DIR, filename))
            df['date'] = pd.to_datetime(df['date'])
            df = df.dropna(subset=FEATURE_COLS).set_index('date')
            if len(df) >= SEQ_LEN:
                symbols.append(sym)
                raw_dfs[sym] = df.iloc[-SEQ_LEN:]
                
    if not symbols: return None, None, None
        
    num_assets = len(symbols)
    panel_features = np.zeros((SEQ_LEN, num_assets, len(FEATURE_COLS)))
    current_prices = np.zeros(num_assets)
    
    for i, sym in enumerate(symbols):
        df_sym = raw_dfs[sym]
        panel_features[:, i, :] = df_sym[FEATURE_COLS].values
        current_prices[i] = df_sym['close'].iloc[-1]
        
    mean = np.nanmean(panel_features, axis=(0, 1), keepdims=True)
    std = np.nanstd(panel_features, axis=(0, 1), keepdims=True) + 1e-8
    panel_features = (panel_features - mean) / std
    window_x = np.transpose(panel_features, (1, 0, 2))
    tensor_x = torch.tensor(window_x, dtype=torch.float32).unsqueeze(0)
    
    return tensor_x, symbols, current_prices

def main():
    print("\n" + "="*60)
    print(" 🎯 [SkyNet V12 Deep Sniper] 重工业时空执行层启动")
    print("="*60)
    
    tensor_x, symbols, current_prices = get_latest_spatial_temporal_tensor()
    if tensor_x is None:
        print("❌ 特征数据不足，无法构建时空张量！")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatialTemporalTransformer(num_assets=len(symbols)).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    else:
        print("❌ 找不到 ST-Transformer 模型权重文件！")
        return
        
    with torch.no_grad():
        preds = model(tensor_x.to(device)).squeeze(0).cpu().numpy()
        
    t5_raw = preds[:, 2]
    t5_z = (t5_raw - np.mean(t5_raw)) / (np.std(t5_raw) + 1e-8)
    
    portfolio = load_portfolio()
    meta = load_meta()
    universe = load_universe()
    cash = portfolio.get('cash', portfolio.get('financials', {}).get('cash', 0))
    positions = portfolio.get('positions_in_shares', {})
    price_map = {sym: price for sym, price in zip(symbols, current_prices)}

    # --- 核心资产计算 ---
    positions_value = sum([shares * price_map.get(sym, 0) for sym, shares in positions.items()])
    total_value = cash + positions_value
    print(f" 💰 当前现金: ￥{cash:,.2f} | 股票市值: ￥{positions_value:,.2f} | 总资产: ￥{total_value:,.2f}")
    
    results = []
    for i, sym in enumerate(symbols):
        results.append({
            'symbol': sym,
            'name': universe.get(sym, {}).get('name', '未知'),
            'price': current_prices[i],
            't5_z': t5_z[i],
            't5_raw': t5_raw[i]
        })
    results.sort(key=lambda x: x['t5_z'], reverse=True)
    
    actions = []
    
    # --- 卖出防守闭环 (高水位追踪止盈/损 + 趋势破位) ---
    held_symbols = [sym for sym, shares in positions.items() if shares > 0]
    for r in results:
        sym = r['symbol']
        if sym in held_symbols:
            current_price = r['price']
            
            # 初始化或更新高水位元数据
            if sym not in meta:
                meta[sym] = {"highest_price": current_price, "days_held": 0}
            
            meta[sym]["days_held"] += 1
            if current_price > meta[sym]["highest_price"]:
                meta[sym]["highest_price"] = current_price
            
            # 止损条件判定
            highest = meta[sym]["highest_price"]
            drawdown = (current_price - highest) / highest
            
            # A: 高水位回撤触发 (Trailing Stop)
            if drawdown <= -TRAIL_STOP_PCT:
                actions.append(f" 💥 [高水位触发] {r['name']} 从最高点 {highest:.3f} 回撤 {drawdown*100:.2f}% (阈值 {TRAIL_STOP_PCT*100}%)。执行止盈/损！")
            
            # B: 趋势弱化触发 (Z-Score + T+3)
            elif meta[sym]["days_held"] >= 3 and r['t5_z'] <= SELL_Z_TH:
                actions.append(f" 💥 [趋势破位] {r['name']} Z-Score 跌至 {r['t5_z']:.2f} (阈值 {SELL_Z_TH})。执行调仓！")
            
            else:
                actions.append(f" 💤 [继续持有] {r['name']} 状态稳定 (回撤: {drawdown*100:.2f}%, Z: {r['t5_z']:.2f})。")

    # --- 汰弱留强逻辑 (Preemption) ---
    # 如果现金不足以支撑买入 Top-2 标的，寻找表现最差且已过冷却期的持仓进行换仓
    
    # 找出当天的 Top-2 进攻标的
    buy_targets = []
    for r in results:
        if len(buy_targets) >= 2: break
        if r['symbol'] in held_symbols: continue
        if r['t5_z'] >= BUY_Z_TH:
            buy_targets.append(r)
            
    for target in buy_targets:
        # 计算该标的需要的估算子弹
        alloc_pct = 0.40 if target['t5_z'] >= HIGH_CONVICTION_Z_TH else 0.20
        estimated_trade_amount = total_value * alloc_pct
        
        # 如果现金不够，且这笔交易是有意义的 (大于 1500)
        if cash < (estimated_trade_amount + FEE_MIN) and estimated_trade_amount >= MIN_TRADE_AMOUNT:
            # 寻找“替罪羊”：持有超过 3 天，且 Z-Score 最低，且 Z-Score 必须为负 (表现跑输大盘)
            weakest_sym = None
            min_z = 0 # 只有负分才考虑被替换
            
            for sym in held_symbols:
                # 获取该持仓的实时 Z-Score
                sym_r = next((item for item in results if item['symbol'] == sym), None)
                if sym_r and meta.get(sym, {}).get("days_held", 0) >= 3:
                    if sym_r['t5_z'] < min_z:
                        min_z = sym_r['t5_z']
                        weakest_sym = sym
            
            if weakest_sym:
                w_name = universe.get(weakest_sym, {}).get('name', '未知')
                actions.append(f" ⚡ [汰弱留强] 现金不足以买入 {target['name']}，系统强制枪毙表现最差持仓 {w_name} (Z:{min_z:.2f})！腾出子弹执行换仓。")
                # 模拟卖出增加现金 (仅为生成指令，实际成交由大佬执行或后续脚本自动下单)
                w_shares = positions.get(weakest_sym, 0)
                w_price = price_map.get(weakest_sym, 0)
                cash += (w_shares * w_price - FEE_MIN)
                held_symbols.remove(weakest_sym) # 模拟持仓更新，防止被买入逻辑误判

    # --- 进攻打靶闭环 (动态仓位 | Top-2) ---
    buy_count = 0
    for r in results:
        if buy_count >= 2: break
        if r['symbol'] in held_symbols: continue

        trade_amount = 0
        sizing_desc = ""
        if r['t5_z'] >= HIGH_CONVICTION_Z_TH:
            trade_amount = total_value * 0.4
            sizing_desc = f"Z-Score > {HIGH_CONVICTION_Z_TH}，重仓出击(40%)"
        elif r['t5_z'] >= BUY_Z_TH:
            trade_amount = total_value * 0.2
            sizing_desc = f"Z-Score > {BUY_Z_TH}，标准建仓(20%)"
        
        if trade_amount > 0:
            if trade_amount < MIN_TRADE_AMOUNT:
                actions.append(f" 🟡 [交易取消] {r['name']} 计算金额 ￥{trade_amount:.2f} 过小，放弃。")
                continue

            expected_profit = trade_amount * r['t5_raw']
            if expected_profit - (FEE_MIN * 2) <= 0:
                actions.append(f" 🔴 [EV锁死] {r['name']} 预期利润无法覆盖手续费，拒绝。")
                continue

            if cash >= (trade_amount + FEE_MIN):
                shares = int(trade_amount / r['price'])
                cost = shares * r['price']
                actions.append(f" 🟢 [做多开枪] {r['name']} ({sizing_desc})！买入 {shares} 股！")
                cash -= (cost + FEE_MIN)
                buy_count += 1
                # 新买入的标的，初始化元数据
                meta[r['symbol']] = {"highest_price": r['price'], "days_held": 0}
            else:
                actions.append(f" 🟡 [弹药不足] {r['name']} 现金不足以完成 ￥{trade_amount:.2f} 的部署。")
                
    # 保存元数据
    save_meta(meta)
    
    if not actions:
        print(" 💤 结论：今日无动作。")
    else:
        for act in actions: print(act)

if __name__ == '__main__':
    main()
