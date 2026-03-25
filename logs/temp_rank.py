import sys, json, os, math
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
BASE_DIR = os.getcwd()
HISTORY_DIR = os.path.join(BASE_DIR, 'history_data')
UNIVERSE_FILE = os.path.join(BASE_DIR, 'skynet_universe.json')

with open(UNIVERSE_FILE, 'r', encoding='utf-8') as f:
    universe = json.load(f)

symbols = list(universe['assets'].keys())
momentum_data = []

for sym in symbols:
    csv_path = os.path.join(HISTORY_DIR, f'{sym}.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if len(df) > 1:
            window = 5 if len(df) > 5 else len(df)-1
            pct_5d = df['close'].pct_change(window).iloc[-1]
            vol = df['close'].pct_change().std() * math.sqrt(252)
            sharpe = pct_5d / vol if (vol > 0 and not pd.isna(vol)) else 0
            momentum_data.append({
                'symbol': sym,
                'name': universe['assets'][sym]['name'],
                'momentum_5d': pct_5d,
                'vol': vol,
                'sharpe': sharpe
            })

momentum_data.sort(key=lambda x: x['sharpe'], reverse=True)
print("--- V11.1 Pro 近期动能与夏普评分排行 (全表概览) ---")
for i, a in enumerate(momentum_data[:15]):
    print(f"{i+1:02d}. {a['name']:<12} ({a['symbol']}) | 5日动能: {a['momentum_5d']*100:>6.2f}% | 年化波动: {a['vol']*100:>5.2f}% | 进攻夏普: {a['sharpe']:>5.2f}")
print("\n... (以下省略表现较弱/负夏普标的)")
