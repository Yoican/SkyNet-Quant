import os
import json
import pandas as pd
import sys

# 设置终端输出为 utf-8，避免 Windows 乱码
sys.stdout.reconfigure(encoding='utf-8')

def audit():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UNIVERSE_FILE = os.path.join(BASE_DIR, 'skynet_universe.json')
    HISTORY_DIR = os.path.join(BASE_DIR, 'history_data')

    with open(UNIVERSE_FILE, 'r', encoding='utf-8') as f:
        universe = json.load(f)
    symbols = list(universe['assets'].keys())

    print(f"📊 [SkyNet Data Audit] 正在质检 31 支标的的历史数据...\n")
    print(f"{'代码':<8} | {'名称':<12} | {'状态':<6} | {'总行数':<6} | {'起始日期':<10} | {'结束日期':<10} | {'脏数据'}")
    print("-" * 80)

    stats = {"OK": 0, "Missing": 0, "Incomplete": 0, "Dirty": 0}

    for sym in symbols:
        name = universe['assets'][sym]['name']
        csv_path = os.path.join(HISTORY_DIR, f"{sym}.csv")
        
        if not os.path.exists(csv_path):
            print(f"{sym:<8} | {name:<12} | ❌缺失 | {'0':<6} | {'-':<10} | {'-':<10} | -")
            stats["Missing"] += 1
            continue
            
        try:
            df = pd.read_csv(csv_path)
            if df.empty or len(df.columns) < 2:
                print(f"{sym:<8} | {name:<12} | ⚠️空表 | {'0':<6} | {'-':<10} | {'-':<10} | -")
                stats["Incomplete"] += 1
                continue
                
            rows = len(df)
            start_date = str(df['date'].iloc[0])[:10]
            end_date = str(df['date'].iloc[-1])[:10]
            
            # Check for NaNs
            try:
                dirty_count = df[['open', 'close', 'high', 'low', 'vol']].isna().sum().sum()
                # 检查是否有值为 0 的异常数据 (除成交量外)
                zero_count = (df[['open', 'close', 'high', 'low']] == 0).sum().sum()
                total_dirty = dirty_count + zero_count
                dirty_str = f"{total_dirty}处" if total_dirty > 0 else "无"
                if total_dirty > 0:
                    stats["Dirty"] += 1
            except:
                dirty_str = "列异常"
                stats["Dirty"] += 1
            
            if rows < 100:
                status = "⚠️残缺"
                stats["Incomplete"] += 1
            else:
                status = "✅正常"
                stats["OK"] += 1
                
            print(f"{sym:<8} | {name:<12} | {status:<5} | {rows:<6} | {start_date:<10} | {end_date:<10} | {dirty_str}")
            
        except Exception as e:
            print(f"{sym:<8} | {name:<12} | ❌错误 | {str(e)[:10]}")
            stats["Missing"] += 1

    print("-" * 80)
    print(f"📈 质检总结: 完美就绪 {stats['OK']} 支 | 数据残缺 {stats['Incomplete']} 支 | 完全缺失 {stats['Missing']} 支 | 包含异常 {stats['Dirty']} 支")

if __name__ == '__main__':
    audit()