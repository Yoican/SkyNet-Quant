import os
import json
import pandas as pd
import akshare as ak
import time
import datetime
import sys
import random

sys.stdout.reconfigure(encoding='utf-8')

def repair_missing_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UNIVERSE_FILE = os.path.join(BASE_DIR, 'skynet_universe.json')
    HISTORY_DIR = os.path.join(BASE_DIR, 'history_data')

    with open(UNIVERSE_FILE, 'r', encoding='utf-8') as f:
        universe = json.load(f)
    symbols = list(universe['assets'].keys())

    end_date = datetime.datetime.now().strftime("%Y%m%d")
    start_date = "20240101"

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

    print(f"🛠️ [SkyNet Repair] 启动数据修复，目标：{len(missing_symbols)} 支断更标的...")
    
    success_count = 0
    for sym in missing_symbols:
        name = universe['assets'][sym]['name']
        print(f"正在抢修 {sym} ({name})...", end=" ")
        
        success = False
        csv_path = os.path.join(HISTORY_DIR, f"{sym}.csv")
        
        # 策略 1: EM 接口 (慢速重试)
        for attempt in range(2):
            try:
                df = ak.fund_etf_hist_em(symbol=sym, period="daily", start_date=start_date, end_date=end_date, adjust="hfq")
                if not df.empty:
                    df = df[['日期', '开盘', '收盘', '最高', '最低', '成交量']]
                    df.columns = ['date', 'open', 'close', 'high', 'low', 'vol']
                    df.to_csv(csv_path, index=False)
                    success = True
                    print(f"✅ [EM] 修复成功 ({len(df)}行)")
                    break
            except Exception as e:
                time.sleep(2)
                
        # 策略 2: 新浪/腾讯 底层指数降维打击 (如果 EM 失败)
        if not success:
            print(f"⚠️ [EM] 失败，尝试降维打击(底层指数)...", end=" ")
            try:
                prefix = "sh" if sym.startswith("5") else "sz"
                df = ak.stock_zh_index_daily_em(symbol=f"{prefix}{sym}")
                if not df.empty:
                    # 适配列名
                    df = df[['date', 'open', 'close', 'high', 'low', 'volume']]
                    df.columns = ['date', 'open', 'close', 'high', 'low', 'vol']
                    df = df[df['date'] >= "2024-01-01"]
                    df.to_csv(csv_path, index=False)
                    success = True
                    print(f"✅ [Index] 降维修复成功 ({len(df)}行)")
            except Exception as e:
                print(f"❌ [Index] 也失败了: {e}")
                
        if success:
            success_count += 1
            
        # 拟人化休眠
        time.sleep(random.uniform(3.0, 5.0))

    print(f"\n🎯 抢修行动结束: 成功恢复 {success_count}/{len(missing_symbols)} 支标的。")

if __name__ == '__main__':
    repair_missing_data()
