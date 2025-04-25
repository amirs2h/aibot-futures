# position_sizer.py
# مدیریت سرمایه حرفه‌ای با Position Sizing پویا بر اساس Volatility و Equity

import pandas as pd
import numpy as np

class PositionSizer:
    def __init__(self, df, equity_col='equity', risk_per_trade=0.01, atr_window=14, max_leverage=5):
        self.df = df.copy()
        self.equity_col = equity_col
        self.risk_per_trade = risk_per_trade
        self.atr_window = atr_window
        self.max_leverage = max_leverage

    def compute_atr(self):
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_window).mean()
        self.df['atr'] = atr
        return atr

    def compute_position_size(self):
        self.compute_atr()
        equity = self.df[self.equity_col].fillna(method='ffill')
        atr = self.df['atr']

        # مقدار ریسک مجاز به دلار
        dollar_risk = equity * self.risk_per_trade

        # اندازه پوزیشن بر اساس ATR
        position_size = dollar_risk / atr

        # محاسبه لوریج معادل، محدودش کن به max_leverage
        notional_value = position_size * self.df['close']
        leverage = notional_value / equity
        leverage = leverage.clip(upper=self.max_leverage)

        self.df['position_size'] = position_size
        self.df['leverage'] = leverage

        return self.df[['timestamp', 'close', 'atr', 'position_size', 'leverage']]

# طرز استفاده:
# sizer = PositionSizer(df_with_equity)
# sizing_df = sizer.compute_position_size()
# df = df.join(sizing_df.set_index('timestamp'), on='timestamp')
