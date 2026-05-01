# APEX Trend Strategy v1.1
# Donchian Channel Breakout + ATR Stop Loss + RSI + Volume Filter
# Long-only | Kraken Spot | 4h bars
# Built for Marco's APEX AI trading system

import numpy as np
from datetime import datetime
from pandas import DataFrame
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter


class ApexTrendStrategy(IStrategy):
    """
    APEX Trend Strategy v1.1
    ========================
    Logic:
    - Uses 3 Donchian channel lookbacks (20, 55, 100 bars) to detect breakouts
    - Averages the signals into a trend score
    - Filters entries with RSI (avoid overbought) and volume spike confirmation
    - Exits on Donchian breakdown or ATR trailing stop
    - Long-only for Kraken spot trading
    """

    INTERFACE_VERSION = 3
    can_short = False
    timeframe = "4h"

    # ROI table
    minimal_roi = {
        "0": 0.15,
        "48": 0.08,
        "96": 0.04,
        "144": 0.02,
    }

    # Hard fallback stop loss
    stoploss = -0.10

    trailing_stop = False
    use_custom_stoploss = True
    process_only_new_candles = True
    startup_candle_count = 110

    # Hyperopt parameters
    buy_rsi_max = IntParameter(55, 75, default=65, space="buy", optimize=True)
    buy_volume_factor = DecimalParameter(1.0, 2.5, default=1.5, space="buy", optimize=True)
    atr_multiplier = DecimalParameter(1.5, 3.0, default=2.0, space="sell", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Donchian Channels (3 lookbacks)
        for lb in [20, 55, 100]:
            dataframe[f"dc_high_{lb}"] = dataframe["high"].shift(1).rolling(lb).max()
            dataframe[f"dc_low_{lb}"] = dataframe["low"].shift(1).rolling(lb).min()

        # Trend Score
        score = np.zeros(len(dataframe))
        for lb in [20, 55, 100]:
            sig = np.where(
                dataframe["close"] > dataframe[f"dc_high_{lb}"], 1.0,
                np.where(dataframe["close"] < dataframe[f"dc_low_{lb}"], -1.0, 0.0)
            )
            score += sig
        dataframe["trend_score"] = score / 3.0

        # ATR (manual calculation - no talib dependency)
        high_low = dataframe["high"] - dataframe["low"]
        high_close = (dataframe["high"] - dataframe["close"].shift()).abs()
        low_close = (dataframe["low"] - dataframe["close"].shift()).abs()
        true_range = high_low.combine(high_close, max).combine(low_close, max)
        dataframe["atr"] = true_range.rolling(20).mean()

        # RSI (manual calculation)
        delta = dataframe["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)
        dataframe["rsi"] = 100 - (100 / (1 + rs))

        # Volume ratio
        dataframe["volume_ma"] = dataframe["volume"].rolling(20).mean()
        dataframe["volume_ratio"] = dataframe["volume"] / dataframe["volume_ma"].replace(0, 1e-10)

        # Realized volatility
        dataframe["returns"] = dataframe["close"].pct_change()
        dataframe["realized_vol"] = dataframe["returns"].rolling(30).std() * np.sqrt(365 * 6)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe["trend_score"] >= 0.66) &
                (dataframe["rsi"] < self.buy_rsi_max.value) &
                (dataframe["volume_ratio"] >= self.buy_volume_factor.value) &
                (dataframe["atr"] > 0) &
                (dataframe["volume"] > 0)
            ),
            "enter_long"
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (dataframe["trend_score"] <= -0.33),
            "exit_long"
        ] = 1

        return dataframe

    def custom_stoploss(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if dataframe is None or dataframe.empty:
            return self.stoploss

        last_candle = dataframe.iloc[-1]
        atr = last_candle["atr"]

        if atr <= 0 or current_rate <= 0:
            return self.stoploss

        atr_stop = (self.atr_multiplier.value * atr) / current_rate
        return -atr_stop

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag,
        side: str,
        **kwargs
    ) -> bool:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if dataframe is None or dataframe.empty:
            return False

        last_candle = dataframe.iloc[-1]

        # Block entry during extreme volatility
        if last_candle["realized_vol"] > 3.0:
            return False

        return True
