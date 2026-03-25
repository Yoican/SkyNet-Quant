import sys
import math
import pandas as pd
import os
import json

# 设置编码
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_DIR = os.path.join(BASE_DIR, 'history_data')

def analyze_correlation():
    # 目标四个标的
    targets = ["518880", "513730", "513100", "512760"]
    names = ["黄金", "东南亚科技", "纳指100", "芯片"]
    
    data = {}
    for sym in targets:
        csv_path = os.path.join(HISTORY_DIR, f"{sym}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            data[sym] = df['close'].pct_change()
    
    returns_df = pd.DataFrame(data).dropna()
    corr_matrix = returns_df.corr()
    
    print("--- 🛡️ '精锐四杰' 相关性矩阵 (Correlation) ---")
    print(corr_matrix)
    
    # 计算平均相关性
    avg_corr = (corr_matrix.values.sum() - 4) / 12
    print(f"\n平均相关性: {avg_corr:.3f}")
    
    # 黄金与其他标的的平均相关性
    gold_corr = corr_matrix["518880"].drop("518880").mean()
    print(f"黄金与权益类标的相关性: {gold_corr:.3f}")

if __name__ == "__main__":
    analyze_correlation()
