import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(BASE_DIR, "v12_features")

SEQ_LEN = 30
HORIZONS = [1, 3, 5]
FEATURE_COLS = ['ret_1d', 'ret_5d', 'ret_20d', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'vol_20d', 'vol_ratio']

class SpatialTemporalDataset(Dataset):
    def __init__(self, data_dir=FEATURES_DIR, seq_len=SEQ_LEN, horizons=HORIZONS, is_train=True):
        self.seq_len = seq_len
        self.horizons = horizons
        
        print("⏳ [重工业级数据引擎] 正在对齐所有 ETF 的时间轴 (Spatial-Temporal Alignment)...")
        
        # 1. 加载所有数据并获取全局交易日历
        raw_data = {}
        all_dates = set()
        self.symbols = []
        
        for filename in sorted(os.listdir(data_dir)):
            if filename.endswith(".csv"):
                sym = filename.replace('.csv', '')
                df = pd.read_csv(os.path.join(data_dir, filename))
                df['date'] = pd.to_datetime(df['date'])
                df = df.dropna(subset=FEATURE_COLS).set_index('date')
                if len(df) > seq_len + max(horizons):
                    raw_data[sym] = df
                    self.symbols.append(sym)
                    all_dates.update(df.index)
                    
        self.num_assets = len(self.symbols)
        self.calendar = sorted(list(all_dates))
        
        # 2. 构建 3D 面板数据 [Total_Dates, Num_Assets, Num_Features]
        panel_shape = (len(self.calendar), self.num_assets, len(FEATURE_COLS))
        panel_close = np.zeros((len(self.calendar), self.num_assets))
        panel_features = np.zeros(panel_shape)
        
        # 填充数据 (处理停牌和缺失值，使用前向填充 ffill)
        for i, sym in enumerate(self.symbols):
            # 将该资产数据 reindex 到全局日历
            df_sym = raw_data[sym].reindex(self.calendar)
            df_sym['close'] = df_sym['close'].ffill().bfill() # 价格前向填充
            for col in FEATURE_COLS:
                df_sym[col] = df_sym[col].ffill().bfill()     # 特征前向填充
                
            panel_close[:, i] = df_sym['close'].values
            panel_features[:, i, :] = df_sym[FEATURE_COLS].values
            
        # 3. 全局 Z-Score 归一化 (防梯度爆炸)
        self.mean = np.nanmean(panel_features, axis=(0, 1), keepdims=True)
        self.std = np.nanstd(panel_features, axis=(0, 1), keepdims=True) + 1e-8
        panel_features = (panel_features - self.mean) / self.std
        
        # 4. 切片构建样本 [Batch, Num_Assets, Seq_Len, Num_Features]
        self.X, self.y = [], []
        
        split_idx = int(len(self.calendar) * 0.8)
        start_idx = 0 if is_train else split_idx
        end_idx = split_idx if is_train else len(self.calendar)
        
        for i in range(start_idx, end_idx - self.seq_len - max(self.horizons)):
            # X shape: [Num_Assets, Seq_Len, Num_Features]
            # 我们需要把时间维度切出来，并转置以符合常规注意力直觉，但这里保持 [Assets, Time, Features] 方便处理
            window_x = panel_features[i : i + self.seq_len, :, :]
            window_x = np.transpose(window_x, (1, 0, 2)) # -> [Num_Assets, Seq_Len, Num_Features]
            
            # y shape: [Num_Assets, Num_Horizons]
            current_close = panel_close[i + self.seq_len - 1, :]
            window_y = np.zeros((self.num_assets, len(self.horizons)))
            
            for h_idx, h in enumerate(self.horizons):
                future_close = panel_close[i + self.seq_len - 1 + h, :]
                ret = (future_close - current_close) / (current_close + 1e-8)
                # 方案A：横向截面 Z-Score 归一化 (让模型预测相对强弱标准分)
                ret_mean = np.nanmean(ret)
                ret_std = np.nanstd(ret) + 1e-8
                z_score = (ret - ret_mean) / ret_std
                window_y[:, h_idx] = z_score
                
            self.X.append(window_x)
            self.y.append(window_y)
            
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

if __name__ == "__main__":
    print("🚀 [SkyNet V12 ST-Pipeline] 启动严谨的时空数据矩阵对齐...")
    dataset = SpatialTemporalDataset(is_train=True)
    print(f"✅ 对齐资产数量: {dataset.num_assets} 支 ETF")
    print(f"✅ 训练集 X Tensor Shape: {dataset.X.shape}  -> [样本数, 资产数, 时间步, 特征数]")
    print(f"✅ 训练集 Y Tensor Shape: {dataset.y.shape}  -> [样本数, 资产数, 预测视界]")