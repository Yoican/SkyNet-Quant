import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from v12_st_optimizer import evaluate_st_model, FEATURES_DIR, SEQ_LEN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INITIAL_CAPITAL = 10000.0
BULLET_SIZE = 2500.0
FEE_MIN = 5.0
BUY_Z_TH = 1.0
SELL_Z_TH = -0.5

def run_backtest(allowed_universe=None, pool_name="全量 33 支 ETF"):
    print(f"\n🚀 [SkyNet V12 盲测炼狱] 启动！交易池: {pool_name}")
    
    dataset, test_calendar, all_preds = evaluate_st_model()
    if dataset is None:
        print("❌ 模型未加载，无法盲测")
        return

    split_idx = int(len(dataset.calendar) * 0.8)
    
    # 准备真实收盘价
    raw_data = {}
    for sym in dataset.symbols:
        df = pd.read_csv(os.path.join(FEATURES_DIR, f"{sym}.csv"))
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        raw_data[sym] = df['close']
        
    panel_close = pd.DataFrame(raw_data).reindex(dataset.calendar).ffill().bfill().values
    
    cash = INITIAL_CAPITAL
    holdings = {}
    
    # 记录统计指标
    daily_values = []
    trades_history = [] # 记录每一笔交易的盈亏
    
    for day_idx, current_date in enumerate(test_calendar):
        current_prices = panel_close[split_idx + day_idx + SEQ_LEN - 1]
        preds_today = all_preds[day_idx]
        
        # 每日结算当前总市值
        current_val = cash
        for sym, h in holdings.items():
            sym_idx = dataset.symbols.index(sym)
            current_val += h['shares'] * current_prices[sym_idx]
        daily_values.append((current_date, current_val))
        
        symbols_to_sell = []
        
        # --- 卖出逻辑 ---
        for sym, h in holdings.items():
            sym_idx = dataset.symbols.index(sym)
            holdings[sym]['days_held'] += 1
            pred_t5_z = preds_today[sym_idx, 2]
            price = current_prices[sym_idx]
            
            # 极速止损 -5%
            cost_basis = h['cost_basis']
            if (h['shares'] * price - cost_basis) / cost_basis <= -0.05:
                sell_val = h['shares'] * price
                cash += (sell_val - FEE_MIN)
                symbols_to_sell.append(sym)
                profit = (sell_val - FEE_MIN) - cost_basis
                trades_history.append({'sym': sym, 'profit': profit, 'type': 'stop_loss'})
                continue
                
            # 正常阈值卖出
            if holdings[sym]['days_held'] >= 3 and pred_t5_z <= SELL_Z_TH:
                sell_val = h['shares'] * price
                cash += (sell_val - FEE_MIN)
                symbols_to_sell.append(sym)
                profit = (sell_val - FEE_MIN) - cost_basis
                trades_history.append({'sym': sym, 'profit': profit, 'type': 'normal_sell'})

        for sym in symbols_to_sell:
            del holdings[sym]
            
        # --- 买入逻辑 ---
        buy_candidates = []
        for sym_idx, sym in enumerate(dataset.symbols):
            # 如果限制了交易池，且不在池子里，则跳过
            if allowed_universe and sym not in allowed_universe:
                continue
                
            pred_t3_z = preds_today[sym_idx, 1]
            pred_t5_z = preds_today[sym_idx, 2]
            
            if pred_t5_z >= BUY_Z_TH and pred_t3_z > 0:
                buy_candidates.append((sym, sym_idx, pred_t5_z))
                
        buy_candidates.sort(key=lambda x: x[2], reverse=True)
        top_candidates = buy_candidates[:2]
        
        for sym, sym_idx, _ in top_candidates:
            if sym not in holdings and cash >= (BULLET_SIZE + FEE_MIN):
                price = current_prices[sym_idx]
                if pd.isna(price) or price <= 0: continue
                shares = int(BULLET_SIZE / price)
                if shares > 0:
                    cost = shares * price + FEE_MIN
                    cash -= cost
                    holdings[sym] = {'shares': shares, 'days_held': 0, 'cost_basis': cost}

    # 强制平仓最后一天
    last_prices = panel_close[split_idx + len(test_calendar) + SEQ_LEN - 1]
    final_val = cash
    for sym, h in holdings.items():
        sym_idx = dataset.symbols.index(sym)
        sell_val = h['shares'] * last_prices[sym_idx]
        final_val += (sell_val - FEE_MIN)
        profit = (sell_val - FEE_MIN) - h['cost_basis']
        trades_history.append({'sym': sym, 'profit': profit, 'type': 'end_of_test'})
        
    daily_values.append((test_calendar[-1], final_val))
    
    # --- 计算统计指标 ---
    df_val = pd.DataFrame(daily_values, columns=['date', 'value']).set_index('date')
    df_val['pct_change'] = df_val['value'].pct_change().fillna(0)
    
    total_return = (final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    # 最大回撤
    roll_max = df_val['value'].cummax()
    drawdown = df_val['value'] / roll_max - 1.0
    max_drawdown = drawdown.min()
    
    # 胜率
    winning_trades = [t for t in trades_history if t['profit'] > 0]
    win_rate = len(winning_trades) / len(trades_history) if trades_history else 0
    
    print("\n" + "="*50)
    print(f" 🏆 V12 盲测战报 ({pool_name})")
    print("="*50)
    print(f" 📅 盲测天数: {len(test_calendar)} 个交易日")
    print(f" 💰 初始本金: ￥{INITIAL_CAPITAL:.2f}")
    print(f" 💵 最终净值: ￥{final_val:.2f} (已扣除所有 5 元手续费)")
    print(f" 📈 绝对收益率: {total_return*100:.2f}%")
    print(f" 📉 最大回撤: {max_drawdown*100:.2f}% (抗风险能力)")
    print(f" 🎯 交易笔数: {len(trades_history)} 笔")
    print(f" 🏅 胜率 (Win Rate): {win_rate*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    # 选项 A：全量 33 支
    run_backtest(allowed_universe=None, pool_name="全量 33 支混战 (Option A)")
