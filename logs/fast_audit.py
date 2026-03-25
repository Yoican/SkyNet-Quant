import requests
import json
import time
import os

def get_realtime_price(symbol):
    # 根据代码前缀判断市场 (简易版：5/6 开头为 sh, 1/0/3 开头为 sz)
    prefix = 'sh' if symbol.startswith(('5', '6')) else 'sz'
    url = f"https://qt.gtimg.cn/q={prefix}{symbol}"
    
    try:
        # 腾讯接口对国内请求非常友好，通常不需要代理
        # 如果还在报错，说明 TUN 劫持了请求且腾讯屏蔽了代理出口 IP
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.text
            # 格式：v_sh510300="1~沪深300ETF~510300~4.662~..."
            parts = data.split('~')
            if len(parts) > 3:
                name = parts[1]
                price = float(parts[3])
                return {"name": name, "price": price, "status": "OK"}
    except Exception as e:
        return {"status": "ERROR", "msg": str(e)}
    return {"status": "NOT_FOUND"}

def fast_audit():
    # [V1.1 Fix] Set encoding to utf-8 for Windows console
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("🐥 [SkyNet Fast Audit] TX Interface Sync Start...")
    
    PRICE_FILE = 'skynet_core/skynet_daily_prices.json'
    UNIVERSE_FILE = 'skynet_core/skynet_universe.json'
    
    if not os.path.exists(UNIVERSE_FILE):
        print("❌ 错误：未找到资产池文件。")
        return

    with open(UNIVERSE_FILE, 'r', encoding='utf-8') as f:
        universe = json.load(f)
    
    # 获取旧价格数据
    if os.path.exists(PRICE_FILE):
        with open(PRICE_FILE, 'r', encoding='utf-8') as f:
            prices = json.load(f)
    else:
        prices = {}

    symbols = list(universe['assets'].keys())
    today_str = "2026-03-18"
    
    success_count = 0
    for sym in symbols:
        print(f"Syncing {sym}...", end=" ", flush=True)
        res = get_realtime_price(sym)
        if res['status'] == 'OK':
            prices[sym] = {
                "name": res['name'],
                "date": today_str,
                "close": res['price'],
                "status": "AUDITED_TX"
            }
            print(f"OK: {res['price']}")
            success_count += 1
        else:
            print(f"FAILED ({res.get('msg', 'Error')})")
        # Slow sync
        time.sleep(1)

    # 保存更新后的价格
    with open(PRICE_FILE, 'w', encoding='utf-8') as f:
        json.dump(prices, f, indent=4, ensure_ascii=False)
    
    print(f"\n✨ 审计完成！成功同步 {success_count}/{len(symbols)} 个标的。")

if __name__ == "__main__":
    fast_audit()
