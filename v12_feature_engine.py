import os
import pandas as pd
import numpy as np

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_DIR = os.path.join(BASE_DIR, "history_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "v12_features")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def calc_rsi(series, period=14):
    """计算相对强弱指数 (RSI)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    # 处理被零除的情况
    rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0)
    return 100 - (100 / (1 + rs))

def process_asset(filepath):
    """清洗单支 ETF 的日线数据并提取多维特征"""
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return None

    # 需要开高低收量全齐，如果缺失太多直接抛弃
    required_cols = ['date', 'open', 'high', 'low', 'close', 'vol']
    if not all(col in df.columns for col in required_cols):
        return None

    if len(df) < 30:
        return None # 数据太短，连 20 日均线都算不出
    
    # 确保按时间正序排列
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # ==========================
    # 🌟 特征工程区 (Feature Engineering)
    # ==========================
    
    # 1. 动量特征 (Momentum) - 继承 V11 并扩展
    df['ret_1d'] = df['close'].pct_change(1)
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_20d'] = df['close'].pct_change(20)
    
    # 2. 微观防骗炮因子：RSI (14日)
    # RSI > 70 超买（警惕砸盘），RSI < 30 超卖（寻找黄金坑）
    df['rsi_14'] = calc_rsi(df['close'], 14)
    
    # 3. 趋势强度因子：MACD (12, 26, 9)
    # 用于捕捉中期波段拐点
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal'] # 能量柱
    
    # 4. 波动率惩罚因子 (20日年化波动率)
    # 波动率突然放大的标的，夏普分要打折
    df['vol_20d'] = df['ret_1d'].rolling(20).std() * np.sqrt(252)
    
    # 5. 量价背离因子 (Volume Proxy)
    # 缩量上涨往往是诱多，放量下跌可能是恐慌盘涌出
    df['vol_sma_5'] = df['vol'].rolling(5).mean()
    df['vol_ratio'] = df['vol'] / (df['vol_sma_5'] + 1e-9) # 当日成交量与近5日均量比值
    
    # 清理因计算滚动窗口产生的 NaN（前面二三十天的数据报废）
    df.dropna(inplace=True)
    return df

def main():
    print("🚀 [SkyNet V12 Feature Engine] 启动！开始铸造高维特征矩阵...")
    if not os.path.exists(HISTORY_DIR):
        print(f"❌ 找不到历史数据目录: {HISTORY_DIR}")
        return

    processed_count = 0
    for filename in os.listdir(HISTORY_DIR):
        if filename.endswith(".csv"):
            filepath = os.path.join(HISTORY_DIR, filename)
            df_features = process_asset(filepath)
            
            if df_features is not None and not df_features.empty:
                out_path = os.path.join(OUTPUT_DIR, filename)
                # 保留两位小数，减小文件体积
                df_features = df_features.round(4)
                df_features.to_csv(out_path, index=False)
                processed_count += 1
                print(f"✔️ 成功提取特征: {filename} (剩余有效天数: {len(df_features)})")
            else:
                print(f"⚠️ 跳过/数据不足: {filename}")
                
    print(f"\n✅ 完毕！共成功提炼 {processed_count} 个标的的多维特征！")
    print(f"📂 特征矩阵已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()