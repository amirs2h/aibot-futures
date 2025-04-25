# hybrid_futures_backtest.py

import pandas as pd
import os
import joblib
import numpy as np
from hybrid_model import HybridTrainer

MODEL_DIR = r"C:\Users\Lenovo\Desktop\aibot"

def backtest_futures_with_hybrid(df, features, target='signal',
                                  leverage=2, fee=0.0004,
                                  initial_balance=1000, sl_pct=0.03, tp_pct=0.1,
                                  confidence_threshold=0.6,
                                  use_saved_models=False,
                                  xgb_weight=0.4, lstm_weight=0.6):

    trainer = HybridTrainer(df, features, target, model_dir=MODEL_DIR)
    X_train, X_test, y_train, y_test = trainer.prepare_data()

    if use_saved_models and \
       os.path.exists(os.path.join(MODEL_DIR, "xgb_model.pkl")) and \
       os.path.exists(os.path.join(MODEL_DIR, "lstm_model.pth")) and \
       os.path.exists(os.path.join(MODEL_DIR, "scaler.pkl")):

        print("\U0001F501 Loading saved models and scaler...")
        trainer.load_models()

    else:
        print("\u2699\ufe0f Training models...")
        trainer.train_xgb(X_train, y_train)
        trainer.train_lstm(X_train, y_train)
        trainer.save_scaler()
        print("\u2705 Models saved to disk successfully.")

    # پیش‌بینی و استخراج سیگنال و اعتماد
    hybrid_proba = trainer.predict_proba(X_test, xgb_weight=xgb_weight, lstm_weight=lstm_weight)
    hybrid_signals = np.argmax(hybrid_proba, axis=1)
    hybrid_confidence = np.max(hybrid_proba, axis=1)

    df = df.iloc[-len(hybrid_signals):].copy()
    df['signal'] = hybrid_signals
    df['confidence'] = hybrid_confidence

    equity = initial_balance
    equity_curve = [equity]
    trades = []
    in_position = False
    entry_price = 0
    entry_time = None
    direction = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        signal = row['signal']
        conf = row['confidence']
        price = row['close']

        if not in_position and signal in [0, 2] and conf >= confidence_threshold:
            in_position = True
            entry_price = price
            entry_time = pd.to_datetime(row['timestamp'])
            direction = 1 if signal == 2 else -1

        elif in_position:
            change = (price - entry_price) / entry_price * direction
            stop_hit = change <= -sl_pct
            take_hit = change >= tp_pct
            exit_signal = (signal != (2 if direction == 1 else 0)) and conf >= confidence_threshold

            if stop_hit or take_hit or exit_signal:
                ret = change * leverage - 2 * fee
                equity *= (1 + ret)

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': pd.to_datetime(row['timestamp']),
                    'direction': 'Long' if direction == 1 else 'Short',
                    'entry_price': entry_price,
                    'exit_price': price,
                    'profit_pct': ret,
                    'confidence': conf
                })

                in_position = False

        equity_curve.append(equity)

    df['equity'] = pd.Series(equity_curve[:len(df)], index=df.index[:len(equity_curve)])

    trades_df = pd.DataFrame(trades)
    final_eq = df['equity'].iloc[-1]
    roi = (final_eq - initial_balance) / initial_balance * 100

    print(f"\n\U0001F4B0 Final Equity: ${final_eq:.2f} | ROI: {roi:.2f}%")

    if not trades_df.empty:
        daily_trades = trades_df.groupby(trades_df['entry_time'].dt.date).size()
        avg_trades_day = daily_trades.mean()
        avg_daily_return = trades_df['profit_pct'].groupby(trades_df['entry_time'].dt.date).sum().mean()

        print(f"\U0001F4C8 Avg trades per day: {avg_trades_day:.2f}")
        print(f"\U0001F4CA Avg daily return: {avg_daily_return * 100:.2f}%")
        print("\n\U0001F4CB Sample trades:")
        print(trades_df.head())

    df[['timestamp', 'close', 'signal', 'confidence', 'equity']].to_csv("hybrid_futures_result.csv", index=False)
    trades_df.to_csv("hybrid_futures_trades.csv", index=False)
    return df, trades_df
