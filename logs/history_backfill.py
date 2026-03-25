import akshare as ak
import pandas as pd
import json
import os
import time
import datetime
import sys

# 设置编码
sys.stdout.reconfigure(encoding='utf-8')

def backfill_history():
    print("🚀 [SkyNet History Engine] 启动全量数据回填 (2025-2026)...")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UNIVERSE_FILE = os.path.join(BASE_DIR, 'skynet_universe.json')
    HISTORY_DIR = os.path.join(BASE_DIR, 'history_data')
    
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)
        
    with open(UNIVERSE_FILE, 'r', encoding='utf-8') as f:
        universe = json.load(f)
    symbols = list(universe['assets'].keys())

    end_date = datetime.datetime.now().strftime("%Y%m%d")
    
    success_count = 0
    for i, symbol in enumerate(symbols):
        csv_path = os.path.join(HISTORY_DIR, f"{symbol}.csv")
        
        # 检查是否已有数据
        if os.path.exists(csv_path):
            try:
                check_df = pd.read_csv(csv_path)
                if len(check_df) > 100:
                    print(f"[{i+1}/{len(symbols)}] ✅ {symbol} ({universe['assets'][symbol]['name']}) 已有历史数据，跳过。")
                    success_count += 1
                    continue
            except:
                pass

        print(f"[{i+1}/{len(symbols)}] 正在同步 {symbol} ({universe['assets'][symbol]['name']})...")
        for attempt in range(3):
            try:
                # 换用 efp 接口可能更稳
                df = ak.fund_etf_hist_em(symbol=symbol, period="daily", start_date="20240101", end_date=end_date, adjust="hfq")
                if not df.empty:
                    df = df[['日期', '开盘', '收盘', '最高', '最低', '成交量']]
                    df.columns = ['date', 'open', 'close', 'high', 'low', 'vol']
                    df.to_csv(csv_path, index=False)
                    success_count += 1
                    print(f"   ✨ {symbol} 同步成功。")
                    time.sleep(5.0)
                    break
            except Exception as e:
                print(f"   ❌ {symbol} 尝试 {attempt+1} 失败: {e}")
                time.sleep(10)
    
    print(f"\n✅ 同步完成！当前资产池历史数据状态: {success_count}/{len(symbols)}")

if __name__ == "__main__":
    backfill_history()
