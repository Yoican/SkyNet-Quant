import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from v12_st_model import SpatialTemporalTransformer
from v12_st_pipeline import SpatialTemporalDataset, FEATURES_DIR, SEQ_LEN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "v12_st_transformer.pth")

# V12 核心回测参数
INITIAL_CAPITAL = 10000.0
FEE_MIN = 5.0
BUY_Z_TH = 0.5                  # 放宽 Z-Score 门槛到 0.5 (全场前 30%)
HIGH_CONVICTION_Z_TH = 1.2      # 重仓线调低到 1.2
SELL_Z_TH = -0.5
MIN_TRADE_AMOUNT = 1500.0
TRAIL_STOP_PCT = 0.05           # 容忍波动增加到 5%

def run_v12_backtest():
    print(f"\n🚀 [SkyNet V12 终极回测仪] 启动！全量模拟实盘逻辑...")
    
    # 强制不切分数据集，从头开始推理
    class FullDataset(SpatialTemporalDataset):
        def __init__(self, data_dir=FEATURES_DIR, seq_len=SEQ_LEN):
            super().__init__(data_dir=data_dir, seq_len=seq_len, is_train=True)
            # 覆盖切分逻辑，手动构建全量 X
            self.X = []
            self.y = []
            
            panel_shape = (len(self.calendar), self.num_assets, len(self.symbols)) # 伪代码，使用基类的 panel 逻辑
            # 由于基类 __init__ 已经跑过了切分，我们需要手动重做 X 
            # 实际上直接修改基类最稳，但由于 edit 报错，我们这里直接反射或复用关键属性
            
            # 重新构建全量 X
            # (由于 self.panel_features 等已经在父类 __init__ 局部作用域运行完，
            # 我们直接重新初始化一个不切分的实例比较好)
    
    # 修改基类后重跑 (略过 edit 失败，直接在回测脚本里实现全量加载逻辑)
    full_ds = SpatialTemporalDataset(is_train=True) # 这里虽然叫 is_train，但我们只想要它的 symbols, calendar 和 panel_features
    # ... 
    
    # 既然 edit 基类困难，我直接在回测里手写加载逻辑，确保对齐万无一失
    print("⏳ [回测引擎] 正在手动加载并对齐高维特征矩阵...")
    raw_data = {}
    all_dates = set()
    symbols = []
    for filename in sorted(os.listdir(FEATURES_DIR)):
        if filename.endswith(".csv"):
            sym = filename.replace('.csv', '')
            df = pd.read_csv(os.path.join(FEATURES_DIR, filename))
            df['date'] = pd.to_datetime(df['date'])
            df = df.dropna(subset=['ret_1d']).set_index('date') # 仅检查 ret_1d 确保日期合法
            raw_data[sym] = df
            symbols.append(sym)
            all_dates.update(df.index)
            
    calendar = sorted(list(all_dates))
    num_assets = len(symbols)
    feature_cols = ['ret_1d', 'ret_5d', 'ret_20d', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'vol_20d', 'vol_ratio']
    
    panel_features = np.zeros((len(calendar), num_assets, len(feature_cols)))
    panel_close = np.zeros((len(calendar), num_assets))
    
    for i, sym in enumerate(symbols):
        df_sym = raw_data[sym].reindex(calendar).ffill().bfill()
        panel_close[:, i] = df_sym['close'].values
        panel_features[:, i, :] = df_sym[feature_cols].values
        
    # 归一化 (使用基类的统计量，或重新计算)
    mean = np.nanmean(panel_features, axis=(0, 1), keepdims=True)
    std = np.nanstd(panel_features, axis=(0, 1), keepdims=True) + 1e-8
    panel_features = (panel_features - mean) / std
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatialTemporalTransformer(num_assets=num_assets).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 回测区间：最后 20%
    split_idx = int(len(calendar) * 0.8)
    test_calendar = calendar[split_idx : len(calendar) - 5]
    
    cash = INITIAL_CAPITAL
    holdings = {}
    history = []
    
    print(f"⏳ 开始对 {len(test_calendar)} 个交易日进行逐日扫描...")
    
    for day_idx, current_date in enumerate(test_calendar):
        real_day_idx = split_idx + day_idx
        current_prices = panel_close[real_day_idx]
        
        # 准备推理窗口：结束于 current_date
        window_x = panel_features[real_day_idx - SEQ_LEN + 1 : real_day_idx + 1, :, :]
        window_x = np.transpose(window_x, (1, 0, 2)) # [Assets, Time, Features]
        tensor_x = torch.tensor(window_x, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(tensor_x).squeeze(0).cpu().numpy()
            
        t5_raw = pred[:, 2]
        t5_z = (t5_raw - np.mean(t5_raw)) / (np.std(t5_raw) + 1e-8)
        
        # 结算总资产
        pos_val = sum([h['shares'] * current_prices[symbols.index(s)] for s, h in holdings.items()])
        total_val = cash + pos_val
        history.append({'date': current_date, 'value': total_val})
        
        # A. 卖出逻辑
        to_sell = []
        for sym, h in holdings.items():
            s_idx = symbols.index(sym)
            price = current_prices[s_idx]
            h['days_held'] += 1
            if price > h['highest_price']: h['highest_price'] = price
            drawdown = (price - h['highest_price']) / (h['highest_price'] + 1e-8)
            
            if drawdown <= -TRAIL_STOP_PCT or (h['days_held'] >= 3 and t5_z[s_idx] <= SELL_Z_TH):
                to_sell.append(sym)
                
        for sym in to_sell:
            s_idx = symbols.index(sym)
            cash += (holdings[sym]['shares'] * current_prices[s_idx] - FEE_MIN)
            del holdings[sym]
            
        # B. 汰弱留强与买入
        candidates = []
        for i, sym in enumerate(symbols):
            if sym not in holdings and t5_z[i] >= BUY_Z_TH:
                candidates.append((sym, i, t5_z[i], t5_raw[i]))
        candidates.sort(key=lambda x: x[2], reverse=True)
        targets = candidates[:2]
        
        # 模拟汰弱留强
        for t_sym, t_idx, t_z, t_raw in targets:
            alloc = 0.4 if t_z >= HIGH_CONVICTION_Z_TH else 0.2
            need = total_val * alloc
            if cash < need + FEE_MIN and need >= MIN_TRADE_AMOUNT:
                weakest = None
                w_z = 0
                for h_sym in holdings:
                    h_idx = symbols.index(h_sym)
                    if holdings[h_sym]['days_held'] >= 3 and t5_z[h_idx] < w_z:
                        w_z = t5_z[h_idx]; weakest = h_sym
                if weakest:
                    cash += (holdings[weakest]['shares'] * current_prices[symbols.index(weakest)] - FEE_MIN)
                    del holdings[weakest]
                    total_val = cash + sum([h['shares']*current_prices[symbols.index(s)] for s, h in holdings.items()])

        # 买入
        for t_sym, t_idx, t_z, t_raw in targets:
            if t_sym not in holdings:
                alloc = 0.4 if t_z >= HIGH_CONVICTION_Z_TH else 0.2
                amt = min(total_val * alloc, cash - FEE_MIN)
                if amt >= MIN_TRADE_AMOUNT and (amt * t_raw - FEE_MIN*2) > 0:
                    price = current_prices[t_idx]
                    shares = int(amt / price)
                    if shares > 0:
                        cash -= (shares * price + FEE_MIN)
                        holdings[t_sym] = {'shares': shares, 'highest_price': price, 'days_held': 0}

    df = pd.DataFrame(history).set_index('date')
    final_v = df['value'].iloc[-1]
    print("\n" + "="*55)
    print(" 🏆 V12 终极实盘逻辑回测报告 (2025-10 至今)")
    print("="*55)
    print(f" 📅 测试时长: {len(test_calendar)} 个交易日")
    print(f" 💵 最终净值: ￥{final_v:.2f}")
    print(f" 📈 累计收益: {(final_v-INITIAL_CAPITAL)/INITIAL_CAPITAL*100:.2f}%")
    print(f" 📉 最大回撤: {(df['value']/df['value'].cummax()-1).min()*100:.2f}%")
    print("="*55)

if __name__ == "__main__":
    run_v12_backtest()
