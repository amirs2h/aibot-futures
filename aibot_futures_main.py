# aibot_futures_main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from hybrid_futures_backtest import backtest_futures_with_hybrid
import os

# مسیر فایل CSV
csv_path = r"C:\Users\Lenovo\Desktop\aibot\btc_15m_data_2018_to_2025.csv"

def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    df.rename(columns={'Open time': 'timestamp', 'Open': 'open', 'High': 'high',
                       'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    # افزودن ویژگی‌ها
    df['rsi'] = RSIIndicator(df['close']).rsi()
    df['stoch_rsi'] = StochRSIIndicator(df['close']).stochrsi()
    df['cci'] = CCIIndicator(df['high'], df['low'], df['close']).cci()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['ema_12'] = EMAIndicator(df['close'], window=12).ema_indicator()
    df['ema_26'] = EMAIndicator(df['close'], window=26).ema_indicator()
    df['adx'] = ADXIndicator(df['high'], df['low'], df['close']).adx()
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    bb = BollingerBands(df['close'])
    df['bb_bbm'] = bb.bollinger_mavg()
    df['bb_bbh'] = bb.bollinger_hband()
    df['bb_bbl'] = bb.bollinger_lband()
    df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['rolling_mean_14'] = df['close'].rolling(14).mean()
    df['rolling_std_14'] = df['close'].rolling(14).std()
    df['volume_change'] = df['volume'].pct_change()

    # تعریف سیگنال
    df['future_return'] = df['close'].shift(-12) / df['close'] - 1
    df['signal'] = 1
    df.loc[df['future_return'] > 0.03, 'signal'] = 1
    df.loc[df['future_return'] < -0.03, 'signal'] = -1
    df.drop(columns=['future_return'], inplace=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def main():
    df = load_and_prepare(csv_path)

    features = [
        'rsi', 'stoch_rsi', 'cci', 'macd', 'macd_signal',
        'ema_12', 'ema_26', 'adx', 'atr',
        'bb_bbm', 'bb_bbh', 'bb_bbl', 'obv',
        'log_return', 'rolling_mean_14', 'rolling_std_14', 'volume_change'
    ]

    df_result, trades = backtest_futures_with_hybrid(
        df, features,
        target='signal',
        leverage=3, fee=0.0004, initial_balance=1000,
        sl_pct=0.03, tp_pct=0.06,
        confidence_threshold=0.4,
        xgb_weight=0.4,
        lstm_weight=0.6,
        use_saved_models=True
    )

    # ذخیره نمودار
    filename = os.path.basename(csv_path).replace(".csv", "_equity.png")
    df_result.plot(x='timestamp', y='equity', figsize=(12, 5), title='Hybrid Futures Equity Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

if __name__ == "__main__":
    main()
