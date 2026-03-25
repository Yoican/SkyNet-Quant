import os
import sys
import pandas as pd
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("❌ 缺少 PyTorch 深度学习框架！")
    print("👉 请在大佬的终端里运行: pip install torch torchvision torchaudio")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(BASE_DIR, "v12_features")

# 深度学习超参数
SEQ_LEN = 30              # 回溯窗口：看过去 30 天的 K 线序列
PRED_HORIZONS = [1, 3, 5] # 多重预测窗口：同时预测 T+1, T+3, T+5 的收益率

class ETFTimeSeriesDataset(Dataset):
    """标准的 PyTorch 时间序列数据集：将 2D 表格转换为 3D 张量"""
    def __init__(self, data_dir, seq_len=30, horizons=[1, 3, 5], is_train=True):
        self.seq_len = seq_len
        self.horizons = horizons
        self.X = []
        self.y = []
        self.symbols = []
        
        feature_cols = ['ret_1d', 'ret_5d', 'ret_20d', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'vol_20d', 'vol_ratio']
        
        print(f"⏳ 正在将 2D 表格重构为 3D 时序张量 (Seq_Len={seq_len})...")
        
        # 记录全局均值和方差用于 Z-Score 归一化 (深度学习必备，否则梯度爆炸)
        all_features = []
        raw_dfs = {}
        
        for filename in os.listdir(data_dir):
            if filename.endswith(".csv"):
                sym = filename.replace('.csv', '')
                filepath = os.path.join(data_dir, filename)
                df = pd.read_csv(filepath).dropna(subset=feature_cols).reset_index(drop=True)
                if len(df) > seq_len + max(horizons):
                    all_features.append(df[feature_cols].values)
                    raw_dfs[sym] = df
                    
        if not all_features:
            raise ValueError("没有足够的数据来构建序列！")
            
        # 计算全局均值和标准差
        concat_features = np.vstack(all_features)
        self.mean = np.mean(concat_features, axis=0)
        self.std = np.std(concat_features, axis=0) + 1e-8
        
        # 构建滑动窗口 (Sliding Windows)
        for sym, df in raw_dfs.items():
            # 归一化特征
            scaled_features = (df[feature_cols].values - self.mean) / self.std
            close_prices = df['close'].values
            
            # 按时间切分训练集(前80%)和测试集(后20%)
            split_idx = int(len(df) * 0.8)
            if is_train:
                start_idx = 0
                end_idx = split_idx
            else:
                start_idx = split_idx
                end_idx = len(df)
                
            for i in range(start_idx, end_idx - self.seq_len - max(self.horizons)):
                # 提取过去 30 天的 2D 矩阵 [30, 9]
                window_x = scaled_features[i : i + self.seq_len]
                
                # 提取未来的多重预测目标 (Multi-Horizon targets)
                current_close = close_prices[i + self.seq_len - 1]
                window_y = []
                for h in self.horizons:
                    future_close = close_prices[i + self.seq_len - 1 + h]
                    ret = (future_close - current_close) / current_close
                    window_y.append(ret)
                    
                self.X.append(window_x)
                self.y.append(window_y)
                self.symbols.append(sym)
                
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def main():
    print("🚀 [SkyNet V12 Deep Quant] 数据管道初始化测试...")
    
    try:
        train_dataset = ETFTimeSeriesDataset(FEATURES_DIR, seq_len=SEQ_LEN, horizons=PRED_HORIZONS, is_train=True)
        test_dataset = ETFTimeSeriesDataset(FEATURES_DIR, seq_len=SEQ_LEN, horizons=PRED_HORIZONS, is_train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        print(f"\n✅ 成功构建训练集 3D 张量: {train_dataset.X.shape}")
        print(f"   👉 维度解析: [{len(train_dataset)}个样本, {SEQ_LEN}天序列, 9个特征]")
        print(f"✅ 成功构建目标集 2D 张量: {train_dataset.y.shape}")
        print(f"   👉 维度解析: [{len(train_dataset)}个样本, {len(PRED_HORIZONS)}个预测期限 (T+1, T+3, T+5)]")
        print(f"✅ 成功构建测试集样本量: {len(test_dataset)}")
        
        # 抽取一个 Batch 看看
        for batch_x, batch_y in train_loader:
            print("\n🔍 抽检第一个 Batch 的数据装载:")
            print(f"   特征 Batch Shape: {batch_x.shape}")
            print(f"   标签 Batch Shape: {batch_y.shape}")
            break
            
        print("\n💡 结论：深度学习的基础底座已完美点亮！随时可以把大模型（LSTM/Transformer）接上来炼丹！")
        
    except Exception as e:
        print(f"\n❌ 数据构建失败: {e}")

if __name__ == "__main__":
    main()