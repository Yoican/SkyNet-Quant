import os
import json
import pandas as pd
import akshare as ak
import time
import datetime
import sys

sys.stdout.reconfigure(encoding='utf-8')

def repair_with_sina():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UNIVERSE_FILE = os.path.join(BASE_DIR, 'skynet_universe.json')
    HISTORY_DIR = os.path.join(BASE_DIR, 'history_data')

    with open(UNIVERSE_FILE, 'r', encoding='utf-8') as f:
        universe = json.load(f)
    symbols = list(universe['assets'].keys())

    missing_symbols = []
    for sym in symbols:
        csv_path = os.path.join(HISTORY_DIR, f"{sym}.csv")
        if not os.path.exists(csv_path):
            missing_symbols.append(sym)
        else:
            try:
                df = pd.read_csv(csv_path)
                if len(df) < 100:
                    missing_symbols.append(sym)
            except:
                missing_symbols.append(sym)

    print(f"🛠️ [SkyNet Sina Repair] 启动新浪备用线路，目标：{len(missing_symbols)} 支...")
    
    success_count = 0
    for sym in missing_symbols:
        name = universe['assets'][sym]['name']
        print(f"正在通过新浪通道抢修 {sym} ({name})...", end=" ")
        
        csv_path = os.path.join(HISTORY_DIR, f"{sym}.csv")
        prefix = "sh" if sym.startswith("5") else "sz"
        full_symbol = f"{prefix}{sym}"
        
        try:
            # 尝试调用新浪的日线行情
            # 注意：新浪返回的格式和东财有差异，必须转换
            df = ak.fund_etf_hist_sina(symbol=full_symbol)
            if not df.empty:
                # 假设返回字段包含: date, open, high, low, close, volume
                # 需确认真实返回字段
                df.columns = [col.lower() for col in df.columns]
                # 过滤出 2024 年以来的
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df[df['date'] >= pd.to_datetime('2024-01-01')]
                    
                # 统一映射
                if 'volume' in df.columns:
                    df.rename(columns={'volume': 'vol'}, inplace=True)
                
                needed_cols = ['date', 'open', 'close', 'high', 'low', 'vol']
                df = df[[c for c in needed_cols if c in df.columns]]
                
                if len(df) > 10:
                    df.to_csv(csv_path, index=False)
                    success_count += 1
                    print(f"✅ 成功 ({len(df)}行)")
                else:
                     print(f"⚠️ 数据太少")
            else:
                 print(f"⚠️ 空数据")
        except Exception as e:
            print(f"❌ 失败: {e}")
            
        time.sleep(2.0)

    print(f"\n🎯 新浪抢修行动结束: 成功恢复 {success_count}/{len(missing_symbols)} 支标的。")

if __name__ == '__main__':
    repair_with_sina()
