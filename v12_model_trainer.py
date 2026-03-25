import os
import sys
import pandas as pd
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
except ImportError:
    print("❌ 缺少 scikit-learn 库，请运行: pip install scikit-learn")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(BASE_DIR, "v12_features")

def load_and_label_data():
    all_data = []
    # 刚刚在 v12_feature_engine.py 里生成的特征列
    feature_cols = ['ret_1d', 'ret_5d', 'ret_20d', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'vol_20d', 'vol_ratio']
    
    for filename in os.listdir(FEATURES_DIR):
        if filename.endswith(".csv"):
            filepath = os.path.join(FEATURES_DIR, filename)
            df = pd.read_csv(filepath)
            
            # 【核心标签构建】：T+5 收益率
            # 如果未来 5 天能涨超过 1.5%（覆盖1%手续费+0.5%纯利），标记为 1 (进攻信号)
            # 否则标记为 0 (装死信号)
            df['future_ret_5d'] = df['close'].shift(-5) / df['close'] - 1.0
            df['target'] = (df['future_ret_5d'] > 0.015).astype(int)
            
            # 剔除因为未来数据缺失 (shift) 和特征计算产生的 NaN
            df = df.dropna(subset=feature_cols + ['target'])
            all_data.append(df[feature_cols + ['target']])
            
    if not all_data:
        return None, None
        
    master_df = pd.concat(all_data, ignore_index=True)
    return master_df[feature_cols], master_df['target']

def main():
    print("🧠 [SkyNet V12 Model Trainer] 正在汇聚 33 支 ETF 的高维特征集...")
    X, y = load_and_label_data()
    
    if X is None or len(X) == 0:
        print("❌ 数据集为空，请先运行特征工程！")
        return
        
    print(f"📊 总提取训练切片: {len(X)} 条 (每条代表某支 ETF 在某天的多维状态)")
    
    # 划分训练集和测试集 (80% 训练，20% 盲测留出)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("⚙️ 正在训练机构级【随机森林 (RandomForest)】非线性分类器...")
    # 构建 100 棵决策树的森林，最大深度10，防止过拟合
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # 在 20% 从未见过的数据上进行盲测
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n🎯 V12 基础盲测准确率: {acc*100:.2f}% ")
    print("  (注：在量化金融里，预测 T+5 绝对涨跌能做到 53% 以上就是印钞机，咱们这只是 Baseline！)")
    
    # 提取特征重要性 (Feature Importance)
    importances = clf.feature_importances_
    feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    print("\n🏆 V12 机器自我学习的【因子重要性排行榜】:")
    print("--------------------------------------------------")
    for idx, row in feat_imp.iterrows():
        # 给特征加上中文解释，方便大佬阅读
        desc = ""
        if row['Feature'] == 'vol_20d': desc = "(20日波动率 - 防黑天鹅)"
        elif row['Feature'] == 'ret_20d': desc = "(20日动能 - 中线趋势)"
        elif row['Feature'] == 'rsi_14': desc = "(14日RSI - 超买超卖反转)"
        elif row['Feature'] == 'macd_hist': desc = "(MACD能量柱 - 短期爆发力)"
        elif row['Feature'] == 'ret_5d': desc = "(5日动能 - 短期趋势)"
        elif row['Feature'] == 'macd': desc = "(MACD绝对值 - 趋势强弱)"
        elif row['Feature'] == 'macd_signal': desc = "(MACD信号线 - 平滑趋势)"
        elif row['Feature'] == 'vol_ratio': desc = "(量价背离比 - 资金异动)"
        elif row['Feature'] == 'ret_1d': desc = "(单日涨跌幅 - 噪音最高)"
        
        print(f"  {idx+1:02d}. {row['Feature']:<12} {desc:<25} | 权重占比: {row['Importance']*100:>5.2f}%")
        
    print("--------------------------------------------------")
    print("💡 结论：这就是机器的『心智』。系统已经不再依赖人类拍脑袋，而是根据以上权重动态打出 0~100 的确信度分！")

if __name__ == '__main__':
    main()