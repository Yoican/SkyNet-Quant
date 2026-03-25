import requests
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def get_historical_data(code):
    prefix = "sh" if code.startswith("51") or code.startswith("60") else "sz"
    url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?_var=kline_dayqfq&param={prefix}{code},day,,,10,qfq"
    try:
        resp = requests.get(url, timeout=5)
        json_str = resp.text.split("kline_dayqfq=")[1]
        data = json.loads(json_str)
        return data['data'][f'{prefix}{code}'].get('qfqday', data['data'][f'{prefix}{code}'].get('day'))
    except:
        return None

codes = ["518880", "512880", "513030", "513520"]
THRESHOLD = -2.0

print("--- Backtest: Last 5 Days Premarket Open vs Prev Close ---")

for code in codes:
    klines = get_historical_data(code)
    if not klines or len(klines) < 6: continue
    
    print(f"\n--- Code: {code} ---")
    # 只看最近 5 个交易日
    for i in range(len(klines)-5, len(klines)):
        curr_date = klines[i][0]
        curr_open = float(klines[i][1])
        prev_close = float(klines[i-1][2])
        change = round((curr_open - prev_close) / prev_close * 100, 2)
        status = "[ALERT]" if change <= THRESHOLD else "[SAFE]"
        print(f"Date: {curr_date} | PrevClose: {prev_close:.3f} | Open: {curr_open:.3f} | Change: {change}% | {status}")
