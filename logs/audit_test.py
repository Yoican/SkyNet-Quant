import akshare as ak
import json
import sys

def audit_test():
    symbols = ["510300", "512100", "513100", "513500", "518880"]
    results = {}
    print("Testing AkShare with fixed TUN...")
    for sym in symbols:
        try:
            df = ak.fund_etf_hist_em(symbol=sym, period="daily", start_date="20260318", end_date="20260318")
            if not df.empty:
                last_price = df.iloc[-1]['收盘']
                results[sym] = {"price": float(last_price), "status": "OK"}
                print(f"✅ {sym}: {last_price}")
            else:
                results[sym] = {"status": "EMPTY"}
                print(f"⚠️ {sym}: No data")
        except Exception as e:
            results[sym] = {"status": "ERROR", "msg": str(e)}
            print(f"❌ {sym}: {str(e)}")
    
    with open("skynet_core/audit_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    audit_test()
