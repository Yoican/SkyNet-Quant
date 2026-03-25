import sys
import os
import pandas as pd
import numpy as np
import json
import warnings
import datetime
import xgboost as xgb

# 确保能导入 v11_modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if BASE_DIR not in sys.path: sys.path.append(BASE_DIR)
if ROOT_DIR not in sys.path: sys.path.append(ROOT_DIR)

from v11_modules.hmm_regime import HMMMarketRegimeDetector
from v11_modules.xgboost_online import XGBoostOnlineLearner

warnings.filterwarnings('ignore')

def run_pure_v11_brain():
    print("🧠 [SkyNet V11 Pro Brain] 启动纯净核心分析引擎...")
    
    HISTORY_DIR = os.path.join(BASE_DIR, 'history_data')
    UNIVERSE_FILE = os.path.join(BASE_DIR, 'skynet_universe.json')
    
    with open(UNIVERSE_FILE, 'r', encoding='utf-8') as f:
        universe = json.load(f)
    
    symbols = list(universe['assets'].keys())
    results = []
    
    # 1. 寻找具备完整历史数据 (>=60行) 的标的
    ready_symbols = []
    for s in symbols:
        p = os.path.join(HISTORY_DIR, f"{s}.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            if len(df) >= 60:
                ready_symbols.append(s)
    
    print(f"🔍 识别到 {len(ready_symbols)} 支具备分析条件的标的。")

    # XGBoost 参数
    xgb_params = {
        'objective': 'reg:absoluteerror',
        'eta': 0.1,
        'max_depth': 4,
        'nthread': 4,
        'eval_metric': 'mae'
    }

    for sym in ready_symbols:
        csv_path = os.path.join(HISTORY_DIR, f"{sym}.csv")
        df = pd.read_csv(csv_path)
        
        # --- 特征提取 ---
        df['pct_chg_1d'] = df['close'].pct_change()
        df['pct_chg_5d'] = df['close'].pct_change(periods=5)
        df['ma60'] = df['close'].rolling(60).mean()
        df['volatility_10d'] = df['pct_chg_1d'].rolling(10).std()
        df['target'] = df['pct_chg_1d'].shift(-1) # 预测下一日收益
        df_clean = df.dropna()
        
        if len(df_clean) < 20: continue
        
        # --- HMM 市场环境探测 (加入容错) ---
        detector = HMMMarketRegimeDetector()
        train_hmm = df_clean.tail(100)
        try:
            detector.fit(train_hmm['pct_chg_1d'], train_hmm['volatility_10d'])
            regime_label = detector.predict_regime(df_clean['pct_chg_1d'].tail(10), df_clean['volatility_10d'].tail(10))
        except Exception as e:
            regime_label = "Chop" # 失败时默认切入太极防守挡
        
        # --- XGBoost 滚动训练与预测 ---
        learner = XGBoostOnlineLearner(params=xgb_params)
        # 使用最近 252 天数据进行训练
        train_xgb = df_clean.tail(252)
        X_train = train_xgb[['pct_chg_1d', 'pct_chg_5d', 'volatility_10d']]
        y_train = train_xgb['target']
        
        learner.initial_train(X_train, y_train)
        
        # 预测最后一天对应的“明天”
        last_feat = df_clean[['pct_chg_1d', 'pct_chg_5d', 'volatility_10d']].tail(1)
        pred_return = learner.predict(last_feat)[0]
        
        # 计算偏离度
        curr_price = df_clean['close'].iloc[-1]
        bias_ma60 = (curr_price - df_clean['ma60'].iloc[-1]) / df_clean['ma60'].iloc[-1]

        results.append({
            "symbol": sym,
            "name": universe['assets'][sym]['name'],
            "regime": regime_label,
            "pred_1d_return": pred_return,
            "bias_ma60": bias_ma60,
            "momentum_5d": df_clean['pct_chg_5d'].iloc[-1]
        })

    # 排序
    results.sort(key=lambda x: x['pred_1d_return'], reverse=True)
    
    print("\n" + "="*80)
    print(f"{'代码':<8} | {'名称':<10} | {'环境':<6} | {'明日预估':<8} | {'5D动量':<8} | {'MA60偏离'}")
    print("-" * 80)
    for r in results[:10]:
        print(f"{r['symbol']:<10} | {r['name']:<12} | {r['regime']:<8} | {r['pred_1d_return']:>7.2%} | {r['momentum_5d']:>7.2%} | {r['bias_ma60']:>7.2%}")
    print("="*80)

if __name__ == "__main__":
    run_pure_v11_brain()
