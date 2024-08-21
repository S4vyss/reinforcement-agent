import numpy as np
import pandas as pd

train = pd.read_csv("/kaggle/input/binanceusdbtc/train (2).csv")
valid = pd.read_csv("/kaggle/input/binanceusdbtc/valid.csv")
test = pd.read_csv("/kaggle/input/binanceusdbtc/test.csv")


def compute_rsi(close_prices, window=14):
    """
    Compute the Relative Strength Index (RSI) for a given series of close prices.

    Parameters:
    - close_prices (pd.Series): Series of close prices.
    - window (int): Window size for computing RSI. Default is 14.

    Returns:
    - rsi (pd.Series): Series containing RSI values.
    """
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(close_prices, short_window=12, long_window=26, signal_window=9):
    """
    Compute the Moving Average Convergence Divergence (MACD) for a given series of close prices.

    Parameters:
    - close_prices (pd.Series): Series of close prices.
    - short_window (int): Short window size for MACD. Default is 12.
    - long_window (int): Long window size for MACD. Default is 26.
    - signal_window (int): Signal window size for MACD. Default is 9.

    Returns:
    - macd (pd.Series): Series containing MACD values.
    """
    short_ema = close_prices.ewm(span=short_window, min_periods=1).mean()
    long_ema = close_prices.ewm(span=long_window, min_periods=1).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, min_periods=1).mean()
    macd = macd_line - signal_line
    return macd


def detect_bullish_engulfing(df):
    """
    Detect bullish engulfing candlestick pattern in a DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing OHLC data.

    Returns:
    - bullish_engulfing (pd.Series): Series indicating bullish engulfing pattern (1 for True, 0 for False).
    """
    bullish_engulfing = np.zeros(len(df))
    for i in range(1, len(df)):
        if df['open'].iloc[i] < df['close'].iloc[i-1] and df['close'].iloc[i] > df['open'].iloc[i-1] and \
                df['close'].iloc[i] > df['open'].iloc[i] and df['open'].iloc[i] < df['close'].iloc[i-1]:
            bullish_engulfing[i] = 1
    return pd.Series(bullish_engulfing, index=df.index)


def feature_engineering(train, valid, test):
    for key, df in enumerate([train, valid, test]):
        for column in ["open", "high", "low"]:
            df["feature_z_"+column] = df[column] / df["close"] - 1
        df["feature_z_close"] = df["close"] / df["close"].shift(1) - 1
        df["feature_adj_close"] = df["close"] * 0.8535
        df["feature_z_adj_close"] = df["feature_adj_close"] / \
            df["feature_adj_close"].shift(1) - 1

        for column in range(5, 31, 5):
            df["feature_z_d"+str(column)] = (df['feature_adj_close'].rolling(
                window=5, min_periods=1).sum() / 5) / df["feature_adj_close"] - 1

        df['feature_price_spread'] = df['high'] - df['low']
        # Change in close price
        df['feature_price_momentum'] = df['close'].diff()
        # Relative Strength Index (RSI)
        df['feature_rsi'] = compute_rsi(df['close'])
        # 10-period Exponential Moving Average
        df['feature_ema_10'] = df['close'].ewm(span=10, adjust=False).mean()

        # Volume-related features
        df['feature_volume_spread'] = df['high'] - df['low']
        df['feature_volume_momentum'] = df['volume'].diff()   # Change in volume
        # Volume Weighted Average Price
        df['feature_vwap'] = (df['close'] * df['volume']
                            ).cumsum() / df['volume'].cumsum()

        # Candlestick patterns and technical indicators
        df['feature_bullish_engulfing'] = detect_bullish_engulfing(
            df)   # Bullish Engulfing Candlestick Pattern
        # Moving Average Convergence Divergence (MACD)
        df['feature_macd'] = compute_macd(df['close'])

        # Time-related features
        df['feature_hour_of_day'] = df['date_close'].dt.hour
        df['feature_day_of_week'] = df['date_close'].dt.dayofweek

        # Derived features
        df['feature_price_rate_of_change'] = df['close'].pct_change() * 100
        df['feature_volume_rate_of_change'] = df['volume'].pct_change() * 100
        df['feature_price_to_volume_ratio'] = df['close'] / df['volume']

        df["feature_close"] = df["close"]

        df["feature_open"] = df["open"]

        df["feature_high"] = df["high"]

        df["feature_low"] = df["low"]

        df["feature_volume"] = df["volume"]

        df['price_spread'] = df['high'] - df['low']
        df['price_momentum'] = df['close'].diff()   # Change in close price

        df.dropna(inplace=True)
        df.to_csv(f"data{key}.csv", index=False)
