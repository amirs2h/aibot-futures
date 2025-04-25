
# ----------- تنظیمات API و پارامترها -----------
# BINANCE_API_KEY = "3dd772b5042ea5af60e849a6a576730b096a704b97dbb6c4ed8187aa4083b8b1"
# BINANCE_API_SECRET = "bf382889669fc097d56c116b6d592ce9e33326ed0ce18a6fb123b35b3d9dbaaf"
# TELEGRAM_TOKEN = "7651829228:AAEV27256dM1lx1eFo1jAN2d2dtOf7MxT78"
# TELEGRAM_CHAT_ID = "252304668"
#MODEL_DIR = r"C:\\Users\\Lenovo\\Desktop\\aibot"
# import os
# import asyncio
# import pandas as pd
# import numpy as np
# import requests
# from datetime import datetime
# from binance.client import Client
# from binance.enums import *
# from binance.exceptions import BinanceAPIException
# from ta.momentum import RSIIndicator, StochRSIIndicator
# from ta.trend import MACD, EMAIndicator, ADXIndicator, CCIIndicator
# from ta.volatility import BollingerBands, AverageTrueRange
# from ta.volume import OnBalanceVolumeIndicator
# from hybrid_model import HybridTrainer

# # ----------- تنظیمات API و پارامترها -----------
# BINANCE_API_KEY = "3dd772b5042ea5af60e849a6a576730b096a704b97dbb6c4ed8187aa4083b8b1"
# BINANCE_API_SECRET = "bf382889669fc097d56c116b6d592ce9e33326ed0ce18a6fb123b35b3d9dbaaf"
# TELEGRAM_TOKEN = "7651829228:AAEV27256dM1lx1eFo1jAN2d2dtOf7MxT78"
# TELEGRAM_CHAT_ID = "252304668"
# MODEL_DIR = r"C:\\Users\\Lenovo\\Desktop\\aibot"

# SYMBOL = "BTCUSDT"
# INTERVAL = Client.KLINE_INTERVAL_15MINUTE
# LIMIT = 500
# SL_PCT = 0.03
# TP_PCT = 0.06
# RISK_PCT = 0.01
# LEVERAGE = 3

# client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
# client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'

# def send_telegram(message):
#     url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
#     data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
#     try:
#         requests.post(url, data=data)
#     except Exception as e:
#         print(f"⚠️ Telegram send error: {e}")

# def add_features(df):
#     df['rsi'] = RSIIndicator(df['close']).rsi()
#     df['stoch_rsi'] = StochRSIIndicator(df['close']).stochrsi()
#     df['cci'] = CCIIndicator(df['high'], df['low'], df['close']).cci()
#     macd = MACD(df['close'])
#     df['macd'] = macd.macd()
#     df['macd_signal'] = macd.macd_signal()
#     df['ema_12'] = EMAIndicator(df['close'], window=12).ema_indicator()
#     df['ema_26'] = EMAIndicator(df['close'], window=26).ema_indicator()
#     df['adx'] = ADXIndicator(df['high'], df['low'], df['close']).adx()
#     df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
#     bb = BollingerBands(df['close'])
#     df['bb_bbm'] = bb.bollinger_mavg()
#     df['bb_bbh'] = bb.bollinger_hband()
#     df['bb_bbl'] = bb.bollinger_lband()
#     df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
#     df['log_return'] = np.log(df['close'] / df['close'].shift(1))
#     df['rolling_mean_14'] = df['close'].rolling(14).mean()
#     df['rolling_std_14'] = df['close'].rolling(14).std()
#     df['volume_change'] = df['volume'].pct_change()
#     df.dropna(inplace=True)
#     return df

# def fetch_ohlcv():
#     klines = client.futures_klines(symbol=SYMBOL, interval=INTERVAL, limit=LIMIT)
#     df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
#                                        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
#     df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
#     df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
#     df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
#     return add_features(df)

# def calculate_qty(entry_price, balance):
#     risk_amount = balance * RISK_PCT
#     sl_amount = entry_price * SL_PCT
#     qty = (risk_amount / sl_amount) * LEVERAGE
#     return round(qty, 3)

# def place_order(signal, price):
#     try:
#         balance_info = client.futures_account_balance()
#         usdt_balance = next(item for item in balance_info if item['asset'] == 'USDT')
#         balance = float(usdt_balance['balance'])
#         qty = calculate_qty(price, balance)
#         side = SIDE_BUY if signal == 1 else SIDE_SELL
#         opposite = SIDE_SELL if signal == 1 else SIDE_BUY
#         stop_price = round(price * (1 - SL_PCT), 2) if signal == 1 else round(price * (1 + SL_PCT), 2)
#         tp_price = round(price * (1 + TP_PCT), 2) if signal == 1 else round(price * (1 - TP_PCT), 2)

#         client.futures_create_order(symbol=SYMBOL, side=side, type=ORDER_TYPE_MARKET, quantity=qty)

#         client.futures_create_order(symbol=SYMBOL, side=opposite, type=ORDER_TYPE_LIMIT, price=tp_price,
#                                      quantity=qty, timeInForce=TIME_IN_FORCE_GTC, reduceOnly=True)

#         client.futures_create_order(symbol=SYMBOL, side=opposite, type=ORDER_TYPE_STOP, stopPrice=stop_price,
#                                      quantity=qty, timeInForce=TIME_IN_FORCE_GTC, reduceOnly=True)

#         return f"✅ Order Placed. Entry: {price}, TP: {tp_price}, SL: {stop_price}, Qty: {qty}"

#     except BinanceAPIException as e:
#         return f"❌ Order Error: {e.message}"
#     except Exception as ex:
#         return f"❌ Unexpected Error: {str(ex)}"

# def run_bot():
#     try:
#         df = fetch_ohlcv()
#         trainer = HybridTrainer(df, features=[
#             'rsi', 'stoch_rsi', 'cci', 'macd', 'macd_signal',
#             'ema_12', 'ema_26', 'adx', 'atr',
#             'bb_bbm', 'bb_bbh', 'bb_bbl', 'obv',
#             'log_return', 'rolling_mean_14', 'rolling_std_14', 'volume_change'
#         ], target='signal', model_dir=MODEL_DIR)

#         trainer.load_models()
#         X_input = df[trainer.features].values
#         hybrid_proba = trainer.predict_proba(X_input, xgb_weight=0.4, lstm_weight=0.6)
#         signal = np.argmax(hybrid_proba[-1])
#         confidence = np.max(hybrid_proba[-1])

#         if confidence < 0.5:
#             return

#         last_price = df['close'].iloc[-1]
#         msg = place_order(signal, last_price)
#         send_telegram(f"🤖 Signal: {signal}, Confidence: {confidence:.2f}\n{msg}")

#     except Exception as e:
#         send_telegram(f"❌ Error: {str(e)}")

# if __name__ == "__main__":
#     run_bot()


# ----------- تنظیمات API و پارامترها -----------
# BINANCE_API_KEY = "3dd772b5042ea5af60e849a6a576730b096a704b97dbb6c4ed8187aa4083b8b1"
# BINANCE_API_SECRET = "bf382889669fc097d56c116b6d592ce9e33326ed0ce18a6fb123b35b3d9dbaaf"
# TELEGRAM_TOKEN = "7651829228:AAEV27256dM1lx1eFo1jAN2d2dtOf7MxT78"
# TELEGRAM_CHAT_ID = "252304668"
# MODEL_DIR = r"C:\\Users\\Lenovo\\Desktop\\aibot"
import os
import time
import logging
import traceback
import requests
from binance.client import Client
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from hybrid_model import HybridTrainer

# --------- تنظیمات ---------
SYMBOL = 'BTCUSDT'
TIMEFRAME = '15m'
LIMIT = 500
SL_PCT = 0.03
TP_PCT = 0.06
CONFIDENCE_THRESHOLD = 0.6
MODEL_DIR = r"C:\\Users\\Lenovo\\Desktop\\aibot"
TRADE_LOG = "trade_history.csv"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN") or '7651829228:AAEV27256dM1lx1eFo1jAN2d2dtOf7MxT78'  # قرار دادن توکن واقعی
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or '252304668'

API_KEY = os.getenv("BINANCE_API_KEY") or '3dd772b5042ea5af60e849a6a576730b096a704b97dbb6c4ed8187aa4083b8b1'
API_SECRET = os.getenv("BINANCE_API_SECRET") or 'bf382889669fc097d56c116b6d592ce9e33326ed0ce18a6fb123b35b3d9dbaaf'

client = Client(API_KEY, API_SECRET, testnet=True)


def send_telegram(text):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
        requests.post(url, data=data)
    except Exception as e:
        logging.warning(f"تلگرام شکست خورد: {e}")


def fetch_data():
    ohlcv = client.futures_klines(symbol=SYMBOL, interval=TIMEFRAME, limit=LIMIT)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


def add_features(df):
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

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
    df.dropna(inplace=True)
    return df


def predict(df):
    features = [
        'rsi', 'stoch_rsi', 'cci', 'macd', 'macd_signal',
        'ema_12', 'ema_26', 'adx', 'atr',
        'bb_bbm', 'bb_bbh', 'bb_bbl', 'obv',
        'log_return', 'rolling_mean_14', 'rolling_std_14', 'volume_change'
    ]
    trainer = HybridTrainer(df, features, target='signal', model_dir=MODEL_DIR)
    trainer.load_models()
    X = df[features].values
    proba = trainer.predict_proba(X, xgb_weight=0.4, lstm_weight=0.6)
    return proba[-1]


def set_leverage(leverage=3):
    try:
        client.futures_change_leverage(symbol=SYMBOL, leverage=leverage)
        send_telegram(f"✅ لوریج به {leverage}x تغییر کرد.")
    except Exception as e:
        send_telegram(f"❌ خطا در تنظیم لوریج: {e}")


def track_order_status(entry_price, side, qty):
    position_side = 'SELL' if side == 'BUY' else 'BUY'
    while True:
        try:
            positions = client.futures_position_information(symbol=SYMBOL)
            for p in positions:
                if float(p['positionAmt']) == 0:
                    exit_price = float(p['markPrice'])
                    pnl = (exit_price - entry_price) * qty if side == 'BUY' else (entry_price - exit_price) * qty
                    pnl_usdt = pnl * 3  # با لوریج ۳
                    msg = f"✅ پوزیشن بسته شد\nنوع: {side}\nورود: {entry_price:.2f}\nخروج: {exit_price:.2f}\nسود/زیان: {pnl_usdt:.2f} USDT"
                    send_telegram(msg)
                    return
            time.sleep(10)
        except Exception as e:
            logging.error(f"خطا در بررسی خروج از پوزیشن: {e}")
            return


def place_order_with_leverage(signal, confidence, price, leverage):
    set_leverage(leverage)
    side = 'BUY' if signal == 1 else 'SELL'

    try:
        positions = client.futures_position_information(symbol=SYMBOL)
        for pos in positions:
            if pos['symbol'] == SYMBOL and float(pos['positionAmt']) != 0:
                send_telegram("❌ پوزیشن باز دارید. سفارش جدید ارسال نمی‌شود.")
                return
    except Exception as e:
        send_telegram(f"⚠️ خطا در بررسی پوزیشن باز: {e}")
        return

    try:
        order = client.futures_create_order(
            symbol=SYMBOL,
            side=side,
            type='MARKET',
            quantity=0.002
        )

        entry_price = float(client.futures_get_order(symbol=SYMBOL, orderId=order['orderId'])['avgPrice'])
        qty = float(order['origQty'])

        tp_price = entry_price * (1 + TP_PCT) if side == 'BUY' else entry_price * (1 - TP_PCT)
        sl_price = entry_price * (1 - SL_PCT) if side == 'BUY' else entry_price * (1 + SL_PCT)

        client.futures_create_order(
            symbol=SYMBOL,
            side='SELL' if side == 'BUY' else 'BUY',
            type='TAKE_PROFIT_MARKET',
            stopPrice=round(tp_price, 2),
            closePosition=True,
            timeInForce='GTC'
        )

        client.futures_create_order(
            symbol=SYMBOL,
            side='SELL' if side == 'BUY' else 'BUY',
            type='STOP_MARKET',
            stopPrice=round(sl_price, 2),
            closePosition=True,
            timeInForce='GTC'
        )

        send_telegram(f"✅ سفارش {side} با لوریج {leverage}x ثبت شد.\nورود: {entry_price:.2f}\nTP: {tp_price:.2f}\nSL: {sl_price:.2f}")

        track_order_status(entry_price, side, qty)

    except Exception as e:
        send_telegram(f"❌ خطا در ارسال سفارش: {e}")


def run_bot():
    while True:
        try:
            df = fetch_data()
            df = add_features(df)
            signal_arr = predict(df)
            signal = np.argmax(signal_arr)
            confidence = np.max(signal_arr)
            last_price = df['close'].iloc[-1]

            if confidence >= CONFIDENCE_THRESHOLD:
                place_order_with_leverage(signal, confidence, last_price, leverage=3)
            else:
                logging.info(f"🔍 بدون سیگنال قوی | اعتماد: {confidence:.2f}")

        except Exception as e:
            logging.error(traceback.format_exc())
            send_telegram(f"⚠️ خطا در اجرای ربات:\n{e}")

        time.sleep(60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_bot()
