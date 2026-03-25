import akshare as ak
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

# 尝试新浪接口抓取 ETF 日线数据
symbol = "sh512480" # 半导体 ETF
print(f"尝试新浪接口抓取 {symbol} ...")
try:
    df_sina = ak.fund_etf_hist_sina(symbol=symbol)
    print("新浪接口成功:")
    print(df_sina.tail(3))
except Exception as e:
    print(f"新浪接口失败: {e}")

print("\n-----------------------\n")

# 尝试如果直接把 ETF 当作股票用腾讯接口抓日线
print(f"尝试股票接口(新浪)抓取 {symbol} ...")
try:
    df_stock = ak.stock_zh_a_daily(symbol=symbol)
    print("股票接口成功:")
    print(df_stock.tail(3))
except Exception as e:
    print(f"股票接口失败: {e}")

