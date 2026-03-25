import akshare as ak
import pandas as pd
import json
import os
import time
import datetime
import sys

# 设置编码
sys.stdout.reconfigure(encoding='utf-8')

def backfill_robust():
    print("🚀 [SkyNet Robust History] 启动“慢牛”回填计划...")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UNIVERSE_FILE = os.path.join(BASE_DIR, 'skynet_universe.json')
    HISTORY_DIR = os.path.join(BASE_DIR, 'history_data')
    
    with open(UNIVERSE_FILE, 'r', encoding='utf-8') as f:
        universe = json.load(f)
    symbols = list(universe['assets'].keys())

    end_date = datetime.datetime.now().strftime("%Y%m%d")
    
    success_count = 0
    for i, symbol in enumerate(symbols):
        csv_path = os.path.join(HISTORY_DIR, f"{symbol}.csv")
        
        # 严格检查：行数必须 > 100 且列数对齐
        is_ready = False
        if os.path.exists(csv_path):
            try:
                check_df = pd.read_csv(csv_path)
                if len(check_df) > 100 and set(['date','open','close','high','low','vol']).issubset(check_df.columns):
                    print(f"[{i+1}/{len(symbols)}] ✅ {symbol} 数据完整，跳过。")
                    success_count += 1
                    is_ready = True
            except:
                pass
        
        if is_ready: continue

        print(f"[{i+1}/{len(symbols)}] 正在抢修 {symbol} ({universe['assets'][symbol]['name']})...")
        
        # 尝试两个接口：EM (东财) 和 SINA (新浪)
        found = False
        # 1. 尝试 EM (加长休眠)
        try:
            df = ak.fund_etf_hist_em(symbol=symbol, period="daily", start_date="20240101", end_date=end_date, adjust="hfq")
            if not df.empty:
                df = df[['日期', '开盘', '收盘', '最高', '最低', '成交量']]
                df.columns = ['date', 'open', 'close', 'high', 'low', 'vol']
                df.to_csv(csv_path, index=False)
                print(f"   ✨ [EM] 同步成功！")
                found = True
                time.sleep(8.0) # 休息 8 秒，极度克制
        except Exception as e:
            print(f"   ❌ [EM] 失败: {e}")
            time.sleep(5.0)

        # 2. 如果 EM 失败，尝试 SINA (新浪) 备用
        if not found:
            try:
                print(f"   🔄 [SINA] 切换新浪备用接口...")
                # 注意：新浪接口 symbol 需要加 sh/sz 前缀
                full_sym = ("sh" + symbol) if symbol.startswith(('5', '6')) else ("sz" + symbol)
                df = ak.fund_etf_daily_sina(symbol=full_sym) # 新浪日线接口
                if not df.empty:
                    # 新浪接口返回的是全量数据，需要重命名列并截取
                    df.columns = ['date', 'open', 'high', 'low', 'close', 'vol']
                    df = df[['date', 'open', 'close', 'high', 'low', 'vol']]
                    df.to_csv(csv_path, index=False)
                    print(f"   ✨ [SINA] 同步成功！")
                    found = True
                    time.sleep(5.0)
            except Exception as e:
                print(f"   ❌ [SINA] 也失败了: {e}")
        
        if found: success_count += 1

    print(f"\n✅ 全量对齐完成！目前资产池 31 支标的中，共有 {success_count} 支具备完整作战能力。")

if __name__ == "__main__":
    backfill_robust()
