import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from hybrid_model import HybridTrainer as HybridModel
from hybrid_futures_backtest import backtest_futures_with_hybrid
from datetime import timedelta

# تقسیم دیتاست به پنجره‌های متوالی (Walk-forward)
def walk_forward(df, features, target, window_size='365D', step_size='180D'):
    start_date = df['timestamp'].min()
    end_date = df['timestamp'].max()

    results = []
    current_start = start_date

    while current_start + pd.to_timedelta(window_size) < end_date:
        train_end = current_start + pd.to_timedelta(window_size)
        test_end = train_end + pd.to_timedelta(step_size)

        train_data = df[(df['timestamp'] >= current_start) & (df['timestamp'] < train_end)]
        test_data = df[(df['timestamp'] >= train_end) & (df['timestamp'] < test_end)]

        if len(train_data) < 5000 or len(test_data) < 1000:
            break

        print(f"\n🧪 Window: {current_start.date()} to {test_end.date()} ({len(train_data)} train / {len(test_data)} test)")

        hybrid = HybridModel(train_data, features, target)
        hybrid.run()

        # پیش‌بینی روی test_data
        test_data = test_data.copy()
        proba = hybrid.predict_proba(test_data[features])
        pred = np.argmax(proba, axis=1)
        confidence = np.max(proba, axis=1)

        # هم‌راستاسازی طول
        test_data = test_data.iloc[-len(pred):].copy()
        test_data['pred'] = pred
        test_data['confidence'] = confidence

        test_data = test_data[test_data['confidence'] > 0.6]  # فقط معاملات با اعتماد بالا

        if test_data.empty:
            print("⚠️ No confident predictions in this window.")
            win_rate, sharpe, profit_factor, trades = 0, 0, 0, pd.DataFrame()
        else:
            # استفاده از پیش‌بینی‌ها به صورت inplace در test_data
            test_data[target] = test_data['pred']  # موقتی، چون بک‌تست انتظار دارد ستون target وجود داشته باشد
            df_result, trades = backtest_futures_with_hybrid(test_data, features, target)

            returns = trades['profit_pct'] if not trades.empty else pd.Series(dtype=float)
            win_rate = (returns > 0).mean() if not returns.empty else 0
            sharpe = returns.mean() / returns.std() * np.sqrt(252 * 96) if returns.std() > 0 else 0
            profit_factor = returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if not returns[returns < 0].empty else 0

        results.append({
            'train_start': current_start,
            'train_end': train_end,
            'test_end': test_end,
            'trades': len(trades),
            'win_rate': win_rate,
            'sharpe': sharpe,
            'profit_factor': profit_factor,
        })

        current_start += pd.to_timedelta(step_size)

    return pd.DataFrame(results)

# رسم گزارش تصویری و ذخیره HTML
def generate_html_report(results_df):
    import plotly.express as px
    import plotly.io as pio

    fig = px.line(results_df, x='train_start', y='sharpe', title='📈 Sharpe Ratio Over Time')
    fig2 = px.bar(results_df, x='train_start', y='win_rate', title='✅ Win Rate')
    fig3 = px.bar(results_df, x='train_start', y='profit_factor', title='💰 Profit Factor')

    html_content = pio.to_html(fig, full_html=False) + pio.to_html(fig2, full_html=False) + pio.to_html(fig3, full_html=False)
    with open("walkforward_report.html", "w", encoding="utf-8") as f:
        f.write("<h1>📊 Walk-Forward Evaluation Report</h1>" + html_content)
    print("📄 گزارش HTML ذخیره شد: walkforward_report.html")

# اجرای نهایی برای کل پروژه
if __name__ == "__main__":
    from aibot_hybrid_main import load_and_prepare

    csv_path = r"C:\Users\Lenovo\Desktop\aibot\btc_15m_data_2018_to_2025.csv"
    df = load_and_prepare(csv_path)

    features = [
        'rsi', 'stoch_rsi', 'cci', 'macd', 'macd_signal',
        'ema_12', 'ema_26', 'adx', 'atr',
        'bb_bbm', 'bb_bbh', 'bb_bbl', 'obv',
        'log_return', 'rolling_mean_14', 'rolling_std_14', 'volume_change'
    ]

    target = 'signal'
    results_df = walk_forward(df, features, target, window_size='365D', step_size='180D')

    generate_html_report(results_df)
