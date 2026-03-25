import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from v12_st_pipeline import SpatialTemporalDataset, SEQ_LEN, HORIZONS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "v12_st_transformer.pth")

class SpatialTemporalTransformer(nn.Module):
    def __init__(self, num_assets=33, input_dim=9, d_model=64, nhead=4, num_layers=2, out_dim=3):
        super(SpatialTemporalTransformer, self).__init__()
        self.num_assets = num_assets
        
        # 1. 独立时间维度的特征编码 (Temporal Embedding)
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1, SEQ_LEN, d_model)) # [1, 1, Time, Dim]
        
        # 2. 时间自注意力编码器 (捕捉各资产自己的主升浪)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True, dropout=0.1)
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. 跨资产空间注意力机制 (Cross-Asset Spatial Attention)
        # 这一层让 33 支资产互相看着对方的脸色行事 (比如芯片盯紧纳指)
        spatial_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True, dropout=0.1)
        self.spatial_transformer = nn.TransformerEncoder(spatial_layer, num_layers=1) # 1层足够提取截面环境
        
        # 4. 多重视界输出头 (Multi-Horizon Head)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, out_dim)
        )

    def forward(self, x):
        # x shape: [Batch, Assets=33, Time=30, Features=9]
        B, N, T, F = x.shape
        
        # 加入位置编码
        x = self.embedding(x) + self.pos_encoder # [B, N, T, d_model]
        
        # --- 阶段一：时间维度提纯 (Temporal Attention) ---
        # 摊平资产维度，让每支资产独立经过 Transformer 提取 30 天时序特征
        x_temp = x.view(B * N, T, -1) # [B*N, T, d_model]
        x_temp = self.temporal_transformer(x_temp)
        
        # 取时间序列最后一个步长作为该资产当前状态的 Summarization
        temporal_summary = x_temp[:, -1, :] # [B*N, d_model]
        temporal_summary = temporal_summary.view(B, N, -1) # [B, N, d_model]
        
        # --- 阶段二：重工业跨资产融合 (Spatial Cross-Attention) ---
        # 现在的 shape 是 [Batch, 33 支资产, 特征向量]
        # 资产之间开始互相计算 Attention 权重，比如纳指会把权重传导给芯片
        spatial_context = self.spatial_transformer(temporal_summary) # [B, N, d_model]
        
        # --- 阶段三：三维预测输出 ---
        # 针对每个资产独立输出 T+1, T+3, T+5 的预期收益
        out = self.fc_out(spatial_context) # [B, N, out_dim=3]
        return out

def train_model():
    print("🚀 [SkyNet V12 Deep Quant] 启动重工业级【时空交叉图网络】(ST-Transformer) 熔炉...")
    
    # 构建 4D 张量时空对齐数据 (这一步非常吃内存，但咱们 33 个资产完全 hold 住)
    train_dataset = SpatialTemporalDataset(is_train=True)
    test_dataset = SpatialTemporalDataset(is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ 运算引擎: {device.type.upper()} | 目标资产池: {train_dataset.num_assets} 支")
    
    model = SpatialTemporalTransformer(num_assets=train_dataset.num_assets).to(device)
    
    # 损失函数与优化器
    criterion = nn.HuberLoss() # 对极端暴涨暴跌免疫
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epochs = 100
    print(f"\n🔥 全市场级联动炼丹开始！(Epochs: {epochs}, 资产互看注意机制已激活)")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        val_loss /= len(test_loader)
        
        print(f"   Epoch {epoch+1:02d}/{epochs} | 联机训练 Loss: {train_loss:.6f} | 盲测 Loss: {val_loss:.6f}")
        
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n✅ 工业级锻造完成！V12 ST-Transformer 脑图权重已固化至: {MODEL_PATH}")
    print("💡 结语：A股芯片终于学会了盯着美股纳指看！时空穿透，降维打击达成！")

if __name__ == "__main__":
    train_model()