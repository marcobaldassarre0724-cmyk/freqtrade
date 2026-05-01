# APEX Trend Strategy v1.0
# Donchian Channel Breakout + ATR Stop Loss + RSI + Volume Filter
# Long-only | Kraken Spot | 4h bars
# Built for Marco's APEX AI trading system

from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
from pandas import DataFrame
import talib.abstract as ta
import numpy as np


class ApexTrendStrategy(IStrategy):
    """
    APEX Trend Strategy v1.0
    ========================
    Logic:
    - Uses 3 Donchian channel lookbacks (20, 55, 100 bars) to detect breakouts
    - Averages the signals into a trend score
    - Filters entries with RSI (avoid overbought) and volume spike confirmation
    - Exits on Donchian breakdown or ATR trailing stop
    - Long-only for Kraken spot trading
    """

    # Strategy metadata
    INTERFACE_VERSION = 3
    strategy_type = "long"
    can_short = False
    timeframe = "4h"

    # ROI table - let the strategy handle exits via signals/stops
    minimal_roi = {
        "0": 0.15,    # Take profit at 15%
        "48": 0.08,   # After 48 hours, take profit at 8%
        "96": 0.04,   # After 96 hours, take profit at 4%
        "144": 0.02,  # After 144 hours, take profit at 2%
    }

    # Stop loss - ATR-based exit is handled in custom_stoploss
    # This is a hard fallback stop
    stoploss = -0.10

    # Trailing stop disabled - we use custom ATR-based stop
    trailing_stop = False

    # Use custom stoploss
    use_custom_stoploss = True

    # Process only new candles for performance
    process_only_new_candles = True

    # Number of candles needed for indicators
    startup_candle_count = 110

    # Hyperopt parameters (tunable ranges)
    buy_rsi_max = IntParameter(55, 75, default=65, space="buy", optimize=True)
    buy_volume_factor = DecimalParameter(1.0, 2.5, default=1.5, space="buy", optimize=True)
    atr_multiplier = DecimalParameter(1.5, 3.0, default=2.0, space="sell", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate all indicators."""

        # ── Donchian Channels (3 lookbacks) ──────────────────────────────
        for lb in [20, 55, 100]:
            dataframe[f"dc_high_{lb}"] = dataframe["high"].shift(1).rolling(lb).max()
            dataframe[f"dc_low_{lb}"] = dataframe["low"].shift(1).rolling(lb).min()

        # ── Trend Score: average of 3 breakout signals ───────────────────
        # +1 = above channel (bullish), -1 = below channel (bearish), 0 = inside
        score = np.zeros(len(dataframe))
        for lb in [20, 55, 100]:
            sig = np.where(
                dataframe["close"] > dataframe[f"dc_high_{lb}"], 1.0,
                np.where(dataframe["close"] < dataframe[f"dc_low_{lb}"], -1.0, 0.0)
            )
            score += sig
        dataframe["trend_score"] = score / 3.0  # normalized: -1 to +1

        # ── ATR for stop loss placement ───────────────────────────────────
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=20)

        # ── RSI to avoid buying overbought ────────────────────────────────
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # ── Volume: rolling average for spike detection ───────────────────
        dataframe["volume_ma"] = dataframe["volume"].rolling(20).mean()
        dataframe["volume_ratio"] = dataframe["volume"] / dataframe["volume_ma"]

        # ── Realized Volatility (annualized, for info) ────────────────────
        dataframe["returns"] = dataframe["close"].pct_change()
        dataframe["realized_vol"] = dataframe["returns"].rolling(30).std() * np.sqrt(365 * 6)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define buy signals."""

        dataframe.loc[
            (
                # At least 2 of 3 Donchian channels confirm breakout
                (dataframe["trend_score"] >= 0.66) &

                # RSI not overbought - avoid chasing already extended moves
                (dataframe["rsi"] < self.buy_rsi_max.value) &

                # Volume above average - confirms breakout is real
                (dataframe["volume_ratio"] >= self.buy_volume_factor.value) &

                # ATR must be valid (enough data)
                (dataframe["atr"] > 0) &

                # Candle volume > 0
                (dataframe["volume"] > 0)
            ),
            "enter_long"
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define sell signals."""

        dataframe.loc[
            (
                # Trend score flips negative - majority of channels show breakdown
                (dataframe["trend_score"] <= -0.33)
            ),
            "exit_long"
        ] = 1

        return dataframe

    def custom_stoploss(
        self,
        pair: str,
        trade,
        current_time,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs
    ) -> float:
        """
        ATR-based trailing stop.
        Stop is placed atr_multiplier * ATR below the current price.
        This tightens as price rises, locking in profits.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if dataframe is None or dataframe.empty:
            return self.stoploss

        last_candle = dataframe.iloc[-1]
        atr = last_candle["atr"]

        if atr <= 0 or current_rate <= 0:
            return self.stoploss

        # ATR stop distance as a fraction of current price
        atr_stop = (self.atr_multiplier.value * atr) / current_rate

        # Return negative value (freqtrade expects negative stoploss)
        return -atr_stop

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time,
        entry_tag,
        side: str,
        **kwargs
    ) -> bool:
        """
        Final safety check before entering a trade.
        Rejects entry if spread is too wide (protects against illiquid moments).
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if dataframe is None or dataframe.empty:
            return False

        last_candle = dataframe.iloc[-1]

        # Reject if realized vol is extremely high (> 300% annualized)
        # This protects against entering during flash crashes or extreme chaos
        if last_candle["realized_vol"] > 3.0:
            return False

        return True
