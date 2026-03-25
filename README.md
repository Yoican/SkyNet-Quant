# SkyNet Quant v12: 基于 ST-Transformer 与集成学习的自动化量化交易平台

SkyNet Quant 是一个端到端的自动化量化交易系统，专注于解决金融时序数据中的高噪声与非线性特征。本项目通过集成学习与深度学习的混合架构，实现了从多源数据异步同步到动态策略执行的闭环控制。

## 核心架构概览

系统采用三层混合模型架构，以应对复杂多变的市场环境：

1. 市场状态感知层 (HMM)：
利用隐马尔可夫模型 (Hidden Markov Model) 进行市场状态探测，识别震荡、单边等不同行情模式，为下游模型提供状态上下文。

2. 核心预测引擎 (ST-Transformer)：
v12 版本引入了时空 Transformer (Spatio-Temporal Transformer) 架构，这是本系统的核心技术突破：
- 时间维度自注意力 (Temporal Self-Attention)：深度捕获资产历史价格序列的长效时序依赖。
- 跨资产空间注意力 (Cross-Asset Spatial Attention)：挖掘不同投资标的间的风险联动效应与相关性。

3. 在线学习微调层 (XGBoost)：
基于集成学习对预测残差进行补偿，利用实时 OHLCV 数据进行在线训练，提升系统对极端行情波动的鲁棒性。

## 技术亮点

- 异步多源数据管道：
独立开发了基于 Python 的异步数据采集与校验系统，确保多源 OHLCV 数据的实时对齐与完整性校验。

- 交易摩擦优化：
内置了基于阈值的交易成本惩罚机制，在回测中严格模拟滑点与佣金，确保策略具备实盘落地意义。

- 样本外盲测表现：
在 2026 年初的严格样本外盲测中，该系统展现了优秀的收益率与回撤控制能力，验证了 ST-Transformer 架构在捕获非线性 Alpha 方面的有效性。

## 项目结构

- `/core`: 包含 ST-Transformer 与 HMM 模型的核心逻辑。
- `/data_pipeline`: 异步数据同步与清洗脚本。
- `/backtest`: 基于交易摩擦优化的回测引擎。
- `/configs`: 策略参数与环境配置文件。

## 开发环境

- Python 3.9+
- PyTorch (用于深度学习模型构建)
- XGBoost
- Pandas / NumPy (数据处理)
- OpenClaw Agent Framework (用于自动化流转)
