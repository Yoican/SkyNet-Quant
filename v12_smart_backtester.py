import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from v12_st_pipeline import SpatialTemporalDataset, FEATURES_DIR, SEQ_LEN
from v12_st_model import SpatialTemporalTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "v12_st_transformer.pth")

INITIAL_CAPITAL = 10000.0
FEE_MIN = 5.0
MIN_ORDER = 1500.0        # 绝不给券商打工的底线
TRAIL_STOP_PCT = 0.04     # 高水位移动止损(回撤4%卖出)
SELL_Z_TH = -0.5          # 彻底跑输大盘止损线

def run_smart_backtest():
    print(f"\n🚀 [SkyNet V12 智能盲测仪] 启动！搭载四大终极交易逻辑")
    
    dataset = SpatialTemporalDataset(is_train=False)
    split_idx = int(len(dataset.calendar) * 0.8)
    test_calendar = dataset.calendar[split_idx : len(dataset.calendar) - SEQ_LEN - 5]
    
    # 准备推理数据
    all_preds = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatialTemporalTransformer(num_assets=dataset.num_assets).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    else:
        print("❌ 模型未加载")
        return

    with torch.no_grad():
        # 我们需要 test_calendar 对应的预测
        # 对齐逻辑：dataset[i] 的输入窗口结束于 calendar[i + SEQ_LEN - 1]
        # 我们的 test_calendar 从 split_idx 开始
        for i in range(split_idx - (SEQ_LEN - 1), split_idx - (SEQ_LEN - 1) + len(test_calendar)):
            if i < 0 or i >= len(dataset):
                # 补一个零向量防止溢出
                all_preds.append(np.zeros((dataset.num_assets, 3)))
                continue
            x, _ = dataset[i]
            x = x.unsqueeze(0).to(device)
            pred = model(x).squeeze(0).cpu().numpy()
            all_preds.append(pred)
            
    all_preds = np.array(all_preds)
    
    # 准备真实收盘价
    raw_data = {}
    for sym in dataset.symbols:
        df = pd.read_csv(os.path.join(FEATURES_DIR, f"{sym}.csv")).set_index('date')
        raw_data[sym] = df['close']
        
    panel_close = pd.DataFrame(raw_data).reindex(dataset.calendar).ffill().bfill().values
    
    cash = INITIAL_CAPITAL
    holdings = {} # {sym: {'shares', 'cost', 'highest_price', 'days_held'}}
    
    daily_values = []
    trades_history = []
    
    for day_idx, current_date in enumerate(test_calendar):
        current_prices = panel_close[split_idx + day_idx + SEQ_LEN - 1]
        preds_today = all_preds[day_idx]
        
        # 0. 结算当日总资产与更新高水位
        current_total_asset = cash
        for sym, h in holdings.items():
            sym_idx = dataset.symbols.index(sym)
            price = current_prices[sym_idx]
            current_total_asset += h['shares'] * price
            
            # 更新历史最高价 (高水位追踪)
            if price > h['highest_price']:
                h['highest_price'] = price
                
        daily_values.append((current_date, current_total_asset))
        
        symbols_to_sell = []
        
        # 1. 卖出逻辑 (高水位追踪止盈/损 + 趋势破位)
        for sym in list(holdings.keys()):
            h = holdings[sym]
            sym_idx = dataset.symbols.index(sym)
            holdings[sym]['days_held'] += 1
            pred_t5_z = preds_today[sym_idx, 2]
            price = current_prices[sym_idx]
            
            # 触发条件 A：从最高点回撤超过 TRAIL_STOP_PCT
            drawdown_from_peak = (price - h['highest_price']) / h['highest_price']
            
            # 触发条件 B：趋势跑输大盘且度过冷却期
            trend_broken = (holdings[sym]['days_held'] >= 3 and pred_t5_z <= SELL_Z_TH)
            
            if drawdown_from_peak <= -TRAIL_STOP_PCT or trend_broken:
                sell_val = h['shares'] * price
                cash += (sell_val - FEE_MIN)
                symbols_to_sell.append(sym)
                profit = (sell_val - FEE_MIN) - h['cost']
                reason = "Trailing_Stop" if drawdown_from_peak <= -TRAIL_STOP_PCT else "Trend_Broken"
                trades_history.append({'sym': sym, 'profit': profit, 'type': reason})

        for sym in symbols_to_sell:
            del holdings[sym]
            
        # 2. 截面掐尖与买入候选
        buy_candidates = []
        for sym_idx, sym in enumerate(dataset.symbols):
            pred_t3_z = preds_today[sym_idx, 1]
            pred_t5_z = preds_today[sym_idx, 2]
            
            # T+5 跑赢大盘 1.0 个标准差，且 T+3 也在涨
            if pred_t5_z >= 1.0: # 放宽一点测试
                buy_candidates.append((sym, sym_idx, pred_t5_z))
                
        buy_candidates.sort(key=lambda x: x[2], reverse=True)
        top_candidates = buy_candidates[:2]
        
        # DEBUG: 打印预测情况
        if day_idx % 20 == 0:
            print(f"DEBUG {current_date}: candidates={len(buy_candidates)}, topZ={top_candidates[0][2] if top_candidates else 'N/A'}")
        
        # 3. 换仓逻辑 (Preemption) - 汰弱留强
        if top_candidates and cash < MIN_ORDER + FEE_MIN:
            best_sym, best_idx, best_z = top_candidates[0]
            # 如果出现史诗级机会 (Z > 1.5) 但没钱了
            if best_z >= 1.5:
                # 寻找最弱的持仓 (Z < 0 且过了冷却期)
                weakest_sym = None
                weakest_z = 999
                for sym, h in holdings.items():
                    s_idx = dataset.symbols.index(sym)
                    s_z = preds_today[s_idx, 2]
                    if s_z < 0 and h['days_held'] >= 3 and s_z < weakest_z:
                        weakest_sym = sym
                        weakest_z = s_z
                        
                if weakest_sym:
                    # 击毙弱者
                    w_idx = dataset.symbols.index(weakest_sym)
                    sell_val = holdings[weakest_sym]['shares'] * current_prices[w_idx]
                    cash += (sell_val - FEE_MIN)
                    profit = (sell_val - FEE_MIN) - holdings[weakest_sym]['cost']
                    trades_history.append({'sym': weakest_sym, 'profit': profit, 'type': 'Preemption_Sell'})
                    del holdings[weakest_sym]
                    # 更新当前可用总资产估计
                    current_total_asset = cash + sum([h['shares']*current_prices[dataset.symbols.index(s)] for s, h in holdings.items()])

        # 4. 动态仓位买入 (Kelly Sizing)
        for sym, sym_idx, z_score in top_candidates:
            if sym not in holdings:
                # 动态比例：Z > 1.5 给 40%，否则 20%
                alloc_pct = 0.40 if z_score >= 1.5 else 0.20
                target_alloc = current_total_asset * alloc_pct
                
                # 受限于实际可用现金
                actual_alloc = min(target_alloc, cash - FEE_MIN)
                
                # EV 物理门槛：低于 1500 块钱，拒绝给券商打工
                if actual_alloc >= MIN_ORDER:
                    price = current_prices[sym_idx]
                    if pd.isna(price) or price <= 0: continue
                    shares = int(actual_alloc / price)
                    if shares > 0:
                        cost = shares * price + FEE_MIN
                        cash -= cost
                        holdings[sym] = {
                            'shares': shares, 
                            'cost': cost,
                            'highest_price': price,
                            'days_held': 0
                        }
                        trades_history.append({'sym': sym, 'profit': 0, 'type': 'Buy'})

    # 强制平仓最后一天
    last_prices = panel_close[split_idx + len(test_calendar) + SEQ_LEN - 1]
    final_val = cash
    for sym, h in holdings.items():
        sym_idx = dataset.symbols.index(sym)
        sell_val = h['shares'] * last_prices[sym_idx]
        final_val += (sell_val - FEE_MIN)
        profit = (sell_val - FEE_MIN) - h['cost']
        trades_history.append({'sym': sym, 'profit': profit, 'type': 'End_of_Test'})
        
    daily_values.append((test_calendar[-1], final_val))
    
    # 统计指标
    df_val = pd.DataFrame(daily_values, columns=['date', 'value']).set_index('date')
    total_return = (final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    roll_max = df_val['value'].cummax()
    drawdown = df_val['value'] / roll_max - 1.0
    max_drawdown = drawdown.min()
    
    winning_trades = [t for t in trades_history if t['profit'] > 0]
    win_rate = len(winning_trades) / len(trades_history) if trades_history else 0
    
    print("\n" + "="*55)
    print(" 🏆 V12 智能四肢：动态仓位与高水位追踪盲测战报")
    print("="*55)
    print(f" 📅 盲测天数: {len(test_calendar)} 个交易日")
    print(f" 💰 初始本金: ￥{INITIAL_CAPITAL:.2f}")
    print(f" 💵 最终净值: ￥{final_val:.2f} (扣除所有手续费)")
    print(f" 📈 绝对收益率: {total_return*100:.2f}%")
    print(f" 📉 最大回撤: {max_drawdown*100:.2f}%")
    print(f" 🎯 交易笔数: {len(trades_history)} 笔")
    print(f" 🏅 胜率 (Win Rate): {win_rate*100:.2f}%")
    print("="*55)

if __name__ == "__main__":
    run_smart_backtest()