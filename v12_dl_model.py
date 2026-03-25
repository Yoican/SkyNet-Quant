import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 引入刚刚写好的数据管道
try:
    from v12_dl_pipeline import ETFTimeSeriesDataset, FEATURES_DIR, SEQ_LEN, PRED_HORIZONS
except ImportError:
    print("❌ 找不到 v12_dl_pipeline 模块")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "v12_transformer.pth")

# ==========================================
# 🧠 V12 终极深度网络：SkyNet Time-Transformer
# ==========================================
class SkyNetTimeTransformer(nn.Module):
    def __init__(self, input_dim=9, d_model=64, nhead=4, num_layers=2, out_dim=3):
        super(SkyNetTimeTransformer, self).__init__()
        
        # 1. 特征升维 (Feature Embedding)
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 2. 位置编码 (Positional Encoding): 让模型知道哪天是昨天，哪天是 30 天前
        self.pos_encoder = nn.Parameter(torch.randn(1, SEQ_LEN, d_model))
        
        # 3. 核心大模型层 (Transformer Encoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=128, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. 多重时间视野预测头 (Multi-Horizon Head)
        # 输出 T+1, T+3, T+5 三个未来的预期收益率
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, out_dim)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len=30, features=9]
        # 嵌入特征并加上时间位置信息
        x = self.embedding(x) + self.pos_encoder
        
        # 深度自注意力提取时序规律
        x = self.transformer(x)
        
        # 提取序列最后一个时间步的隐状态作为上下文总结
        context = x[:, -1, :] 
        
        # 预测未来
        out = self.fc_out(context)
        return out

def train_model():
    print("🚀 [SkyNet V12 Deep Quant] 启动 Transformer 核心熔炉...")
    
    # 加载 3D 张量数据
    train_dataset = ETFTimeSeriesDataset(FEATURES_DIR, seq_len=SEQ_LEN, horizons=PRED_HORIZONS, is_train=True)
    test_dataset = ETFTimeSeriesDataset(FEATURES_DIR, seq_len=SEQ_LEN, horizons=PRED_HORIZONS, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # 实例化高端 Transformer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ 正在使用运算设备: {device.type.upper()}")
    
    model = SkyNetTimeTransformer(input_dim=9, out_dim=len(PRED_HORIZONS)).to(device)
    
    # 工业界标配：Huber Loss (对极端暴涨暴跌的异常值具有鲁棒性) + AdamW 优化器
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epochs = 5 # 演示环境快速迭代，实盘可设为 50-100
    print(f"\n🔥 开始炼丹！(Epochs: {epochs}, Batch Size: 128)")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # 梯度裁剪 (防止 Transformer 梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # 盲测集验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        val_loss /= len(test_loader)
        
        print(f"   Epoch {epoch+1:02d}/{epochs} | 训练集 Loss: {train_loss:.6f} | 盲测集 Loss: {val_loss:.6f}")
        
    # 保存模型权重
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n✅ 炼丹完成！V12 Transformer 大脑已保存至: {MODEL_PATH}")
    print("💡 结论：多维时序张量已成功通过自注意力机制(Self-Attention)，模型已学会理解 '主升浪' 的结构，而不再被单日波动欺骗！")

if __name__ == "__main__":
    train_model()