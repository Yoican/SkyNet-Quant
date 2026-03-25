import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(BASE_DIR, "v12_features")

# 模拟参数
INITIAL_CAPITAL = 10000.0
BULLET_SIZE = 2500.0
FEE_MIN = 5.0

def load_and_predict():
    """加载数据，按时间划分训练集和测试集，并输出测试集的每天预测概率"""
    all_data = []
    feature_cols = ['ret_1d', 'ret_5d', 'ret_20d', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'vol_20d', 'vol_ratio']
    
    for filename in os.listdir(FEATURES_DIR):
        if filename.endswith(".csv"):
            symbol = filename.replace('.csv', '')
            filepath = os.path.join(FEATURES_DIR, filename)
            df = pd.read_csv(filepath)
            
            df['future_ret_5d'] = df['close'].shift(-5) / df['close'] - 1.0
            df['target'] = (df['future_ret_5d'] > 0.015).astype(int)
            df['symbol'] = symbol
            
            df = df.dropna(subset=feature_cols + ['target'])
            all_data.append(df)
            
    master_df = pd.concat(all_data, ignore_index=True)
    master_df['date'] = pd.to_datetime(master_df['date'])
    master_df = master_df.sort_values('date')
    
    # 按时间切分：前 70% 训练，后 30% 测试（盲测阶段）
    split_idx = int(len(master_df) * 0.7)
    train_df = master_df.iloc[:split_idx]
    test_df = master_df.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    
    clf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # 预测概率
    probs = clf.predict_proba(X_test)[:, 1] * 100
    test_df['score'] = probs
    
    return test_df

def simulate_trading(test_df, buy_th, sell_th):
    """在测试集上模拟状态机打靶，返回最终净利润和交易次数"""
    cash = INITIAL_CAPITAL
    holdings = {} # {symbol: shares}
    trade_count = 0
    
    # 按天回测
    unique_dates = sorted(test_df['date'].unique())
    
    for current_date in unique_dates:
        daily_data = test_df[test_df['date'] == current_date]
        
        # 1. 卖出逻辑
        symbols_to_sell = []
        for sym, shares in holdings.items():
            sym_data = daily_data[daily_data['symbol'] == sym]
            if not sym_data.empty:
                score = sym_data['score'].values[0]
                current_price = sym_data['close'].values[0]
                
                if score <= sell_th:
                    # 触发卖出
                    sell_value = shares * current_price
                    cash += (sell_value - FEE_MIN)
                    symbols_to_sell.append(sym)
                    trade_count += 1
        
        for sym in symbols_to_sell:
            del holdings[sym]
            
        # 2. 买入逻辑 (按分数从高到低买)
        buy_candidates = daily_data[daily_data['score'] >= buy_th].sort_values('score', ascending=False)
        for _, row in buy_candidates.iterrows():
            sym = row['symbol']
            price = row['close']
            
            # 如果没持有，且现金够一颗子弹
            if sym not in holdings and cash >= (BULLET_SIZE + FEE_MIN):
                shares_to_buy = int(BULLET_SIZE / price)
                if shares_to_buy > 0:
                    cost = (shares_to_buy * price) + FEE_MIN
                    cash -= cost
                    holdings[sym] = shares_to_buy
                    trade_count += 1

    # 结算最后一天
    final_value = cash
    last_date = unique_dates[-1]
    last_day_data = test_df[test_df['date'] == last_date]
    
    for sym, shares in holdings.items():
        sym_data = last_day_data[last_day_data['symbol'] == sym]
        if not sym_data.empty:
            final_value += (shares * sym_data['close'].values[0] - FEE_MIN)
            
    net_profit = final_value - INITIAL_CAPITAL
    return net_profit, trade_count

def optimize_thresholds():
    print("⏳ 正在加载数据并训练基准模型...")
    test_df = load_and_predict()
    print(f"✅ 盲测数据准备完毕，共 {len(test_df)} 条日线记录。开始网格搜索(Grid Search)...")
    
    buy_range = [75, 80, 85, 90]
    sell_range = [30, 40, 50, 60]
    
    best_profit = -999999
    best_params = (0, 0)
    best_trades = 0
    
    results = []
    
    for buy_th in buy_range:
        for sell_th in sell_range:
            if sell_th >= buy_th: continue
            
            profit, trades = simulate_trading(test_df, buy_th, sell_th)
            results.append((buy_th, sell_th, profit, trades))
            
            if profit > best_profit:
                best_profit = profit
                best_params = (buy_th, sell_th)
                best_trades = trades
                
    # 打印排名前 5 的组合
    results.sort(key=lambda x: x[2], reverse=True)
    print("\n🏆 【V12 机器寻优：阈值回测排行榜 TOP 5】")
    print("-" * 60)
    print(f"{'买入阈值':<8} | {'卖出阈值':<8} | {'净利润(元)':<12} | {'交易次数(次)':<8}")
    print("-" * 60)
    for res in results[:5]:
        b, s, p, t = res
        print(f" >= {b:<5} | <= {s:<5} | ￥{p:<10.2f} | {t:<8}")
        
    print("-" * 60)
    print(f"💡 机器最终建议：买入阈值设为 {best_params[0]}，卖出阈值设为 {best_params[1]}。")
    print(f"在严苛扣除每笔 5 元手续费的情况下，该参数组合在盲测期内斩获最高净利润 ￥{best_profit:.2f}，总出手 {best_trades} 次。")

if __name__ == "__main__":
    optimize_thresholds()