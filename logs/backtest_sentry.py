import requests
import json
import sys

# 设置输出编码
sys.stdout.reconfigure(encoding='utf-8')

def get_historical_data(code):
    prefix = "sh" if code.startswith("51") or code.startswith("60") else "sz"
    # 获取最近 5 天的数据，方便看这一周的趋势
    url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?_var=kline_dayqfq&param={prefix}{code},day,,,5,qfq"
    try:
        resp = requests.get(url, timeout=5)
        json_str = resp.text.split("kline_dayqfq=")[1]
        data = json.loads(json_str)
        # 腾讯接口对 QFQ 数据的字段名有时是 qfqday 有时是 day
        if 'qfqday' in data['data'][f'{prefix}{code}']:
            klines = data['data'][f'{prefix}{code}']['qfqday']
        else:
            klines = data['data'][f'{prefix}{code}']['day']
        return klines
    except Exception as e:
        return None

codes = ["518880", "512880", "513030", "513520"]
THRESHOLD = -2.0

print("--- Backtest: Monday Open (2026-03-23) vs Friday Close (2026-03-20) ---")

for code in codes:
    klines = get_historical_data(code)
    # klines 的最后两个分别是 周一(03-23) 和 周五(03-20)
    # 格式: [日期, 开盘, 收盘, ...]
    if klines and len(klines) >= 2:
        mon_open = float(klines[-1][1])
        fri_close = float(klines[-2][2])
        change = round((mon_open - fri_close) / fri_close * 100, 2)
        print(f"[{code}] Friday Close: {fri_close}, Monday Open: {mon_open}, Change: {change}%")
        if change <= THRESHOLD:
            print(f"  [ALERT] TRIGGERED! Current Drop: {change}%")
        else:
            print(f"  [SAFE] No alert triggered.")
    else:
        print(f"[{code}] Data missing.")
