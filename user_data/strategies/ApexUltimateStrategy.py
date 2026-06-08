# APEX Ultimate Strategy v1.0
# Full indicator suite with Hyperopt optimization
# Long-only | Spot | Built for Marco's APEX AI trading system

import numpy as np
from datetime import datetime
from pandas import DataFrame
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, BooleanParameter


class ApexUltimateStrategy(IStrategy):
    """
    APEX Ultimate Strategy v1.0
    ============================
    Comprehensive indicator suite including:
    - Donchian Channels (3 lookbacks)
    - EMA Crossovers (3 pairs)
    - MACD
    - ADX
    - RSI + Stochastic RSI
    - Williams %R
    - CCI
    - ROC
    - Bollinger Bands
    - Keltner Channels
    - OBV
    - CMF
    - ATR
    - Candlestick Patterns (6 patterns)
    - ADX Regime Gate (NEW)
    - 200 EMA Trend Filter (NEW)
    - ATR Close Confirmation Buffer (NEW)
    All parameters tunable via Hyperopt
    """

    INTERFACE_VERSION = 3
    can_short = False
    timeframe = '30m'

    minimal_roi = {
        "0": 0.459,
        "145": 0.07,
        "278": 0.034,
        "801": 0
    }

    stoploss = -0.022
    trailing_stop = False
    use_custom_stoploss = False
    use_exit_signal = False
    process_only_new_candles = True
    startup_candle_count = 200

    # ── Donchian Parameters ──────────────────────────────────────────────
    dc_trend_score_entry = DecimalParameter(0.1, 1.0, default=0.33, space='buy', optimize=True)
    dc_trend_score_exit = DecimalParameter(-1.0, -0.1, default=-0.33, space='sell', optimize=True)

    # ── RSI Parameters ───────────────────────────────────────────────────
    buy_rsi_max = IntParameter(40, 80, default=57, space='buy', optimize=True)
    buy_rsi_min = IntParameter(10, 40, default=20, space='buy', optimize=True)
    use_rsi = BooleanParameter(default=True, space='buy', optimize=True)

    # ── Stochastic RSI Parameters ────────────────────────────────────────
    buy_stoch_rsi_max = DecimalParameter(0.3, 0.9, default=0.8, space='buy', optimize=True)
    use_stoch_rsi = BooleanParameter(default=True, space='buy', optimize=True)

    # ── MACD Parameters ──────────────────────────────────────────────────
    use_macd = BooleanParameter(default=True, space='buy', optimize=True)
    use_macd_exit = BooleanParameter(default=True, space='sell', optimize=True)

    # ── EMA Parameters ───────────────────────────────────────────────────
    use_ema_9_21 = BooleanParameter(default=True, space='buy', optimize=True)
    use_ema_21_50 = BooleanParameter(default=False, space='buy', optimize=True)
    use_ema_50_200 = BooleanParameter(default=False, space='buy', optimize=True)

    # ── ADX Parameters ───────────────────────────────────────────────────
    buy_adx_min = IntParameter(15, 40, default=25, space='buy', optimize=True)
    use_adx = BooleanParameter(default=True, space='buy', optimize=True)

    # ── Bollinger Band Parameters ────────────────────────────────────────
    use_bb_breakout = BooleanParameter(default=True, space='buy', optimize=True)
    use_bb_exit = BooleanParameter(default=False, space='sell', optimize=True)

    # ── CCI Parameters ───────────────────────────────────────────────────
    buy_cci_min = IntParameter(-200, 0, default=-100, space='buy', optimize=True)
    use_cci = BooleanParameter(default=False, space='buy', optimize=True)

    # ── Williams %R Parameters ───────────────────────────────────────────
    buy_williams_max = IntParameter(-80, -20, default=-50, space='buy', optimize=True)
    use_williams = BooleanParameter(default=False, space='buy', optimize=True)

    # ── ROC Parameters ───────────────────────────────────────────────────
    buy_roc_min = DecimalParameter(0.0, 5.0, default=0.5, space='buy', optimize=True)
    use_roc = BooleanParameter(default=False, space='buy', optimize=True)

    # ── Volume Parameters ────────────────────────────────────────────────
    buy_volume_factor = DecimalParameter(1.0, 3.0, default=1.228, space='buy', optimize=True)
    use_cmf = BooleanParameter(default=False, space='buy', optimize=True)
    buy_cmf_min = DecimalParameter(-0.5, 0.5, default=0.0, space='buy', optimize=True)
    use_obv_trend = BooleanParameter(default=False, space='buy', optimize=True)

    # ── Candlestick Pattern Parameters ───────────────────────────────────
    use_hammer = BooleanParameter(default=False, space='buy', optimize=True)
    use_engulfing = BooleanParameter(default=False, space='buy', optimize=True)
    use_morning_star = BooleanParameter(default=False, space='buy', optimize=True)
    use_three_white_soldiers = BooleanParameter(default=False, space='buy', optimize=True)
    use_doji_reversal = BooleanParameter(default=False, space='buy', optimize=True)
    use_piercing_line = BooleanParameter(default=False, space='buy', optimize=True)

    # ── ATR Stop Parameters ──────────────────────────────────────────────
    atr_multiplier = DecimalParameter(1.0, 4.0, default=2.569, space='sell', optimize=True)

    # ── Custom Exit Profit Threshold ─────────────────────────────────────
    exit_min_profit = DecimalParameter(0.001, 0.05, default=0.025, space='sell', optimize=True)

    # ── NEW: Regime Gate Parameters ──────────────────────────────────────
    adx_regime_min = IntParameter(15, 40, default=25, space='buy', optimize=True)
    use_adx_regime = BooleanParameter(default=True, space='buy', optimize=True)
    use_ema200_filter = BooleanParameter(default=True, space='buy', optimize=True)
    atr_confirm_mult = DecimalParameter(0.0, 1.5, default=0.5, space='buy', optimize=True)
    use_atr_confirm = BooleanParameter(default=True, space='buy', optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # ── ATR ──────────────────────────────────────────────────────────
        high_low = dataframe['high'] - dataframe['low']
        high_close = (dataframe['high'] - dataframe['close'].shift()).abs()
        low_close = (dataframe['low'] - dataframe['close'].shift()).abs()
        true_range = high_low.combine(high_close, max).combine(low_close, max)
        dataframe['atr'] = true_range.rolling(20).mean()

        # ── RSI ──────────────────────────────────────────────────────────
        delta = dataframe['close'].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)
        dataframe['rsi'] = 100 - (100 / (1 + rs))

        # ── Stochastic RSI ────────────────────────────────────────────────
        rsi_min = dataframe['rsi'].rolling(14).min()
        rsi_max = dataframe['rsi'].rolling(14).max()
        dataframe['stoch_rsi'] = (dataframe['rsi'] - rsi_min) / (rsi_max - rsi_min + 1e-10)

        # ── EMAs ─────────────────────────────────────────────────────────
        dataframe['ema_9'] = dataframe['close'].ewm(span=9).mean()
        dataframe['ema_21'] = dataframe['close'].ewm(span=21).mean()
        dataframe['ema_50'] = dataframe['close'].ewm(span=50).mean()
        dataframe['ema_200'] = dataframe['close'].ewm(span=200).mean()

        # ── MACD ─────────────────────────────────────────────────────────
        ema_12 = dataframe['close'].ewm(span=12).mean()
        ema_26 = dataframe['close'].ewm(span=26).mean()
        dataframe['macd'] = ema_12 - ema_26
        dataframe['macd_signal'] = dataframe['macd'].ewm(span=9).mean()
        dataframe['macd_hist'] = dataframe['macd'] - dataframe['macd_signal']

        # ── ADX ──────────────────────────────────────────────────────────
        plus_dm = dataframe['high'].diff()
        minus_dm = -dataframe['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        atr14 = true_range.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr14.replace(0, 1e-10))
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr14.replace(0, 1e-10))
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
        dataframe['adx'] = dx.rolling(14).mean()
        dataframe['plus_di'] = plus_di
        dataframe['minus_di'] = minus_di

        # ── Bollinger Bands ───────────────────────────────────────────────
        bb_mid = dataframe['close'].rolling(20).mean()
        bb_std = dataframe['close'].rolling(20).std()
        dataframe['bb_upper'] = bb_mid + 2 * bb_std
        dataframe['bb_lower'] = bb_mid - 2 * bb_std
        dataframe['bb_mid'] = bb_mid
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / bb_mid

        # ── Keltner Channels ─────────────────────────────────────────────
        kc_mid = dataframe['close'].ewm(span=20).mean()
        dataframe['kc_upper'] = kc_mid + 2 * dataframe['atr']
        dataframe['kc_lower'] = kc_mid - 2 * dataframe['atr']

        # ── Bollinger Band Squeeze ────────────────────────────────────────
        dataframe['bb_squeeze'] = (
            (dataframe['bb_upper'] < dataframe['kc_upper']) &
            (dataframe['bb_lower'] > dataframe['kc_lower'])
        ).astype(float)

        # ── CCI ───────────────────────────────────────────────────────────
        typical_price = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        mean_dev = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        dataframe['cci'] = (typical_price - typical_price.rolling(20).mean()) / (0.015 * mean_dev + 1e-10)

        # ── Williams %R ───────────────────────────────────────────────────
        highest_high = dataframe['high'].rolling(14).max()
        lowest_low = dataframe['low'].rolling(14).min()
        dataframe['williams_r'] = -100 * (highest_high - dataframe['close']) / (highest_high - lowest_low + 1e-10)

        # ── ROC ───────────────────────────────────────────────────────────
        dataframe['roc'] = dataframe['close'].pct_change(10) * 100

        # ── Volume indicators ─────────────────────────────────────────────
        dataframe['volume_ma'] = dataframe['volume'].rolling(20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_ma'].replace(0, 1e-10)

        # OBV
        obv = (np.sign(dataframe['close'].diff()) * dataframe['volume']).fillna(0).cumsum()
        dataframe['obv'] = obv
        dataframe['obv_ema'] = dataframe['obv'].ewm(span=20).mean()
        dataframe['obv_trend'] = (dataframe['obv'] > dataframe['obv_ema']).astype(float)

        # CMF
        mf_multiplier = ((dataframe['close'] - dataframe['low']) - (dataframe['high'] - dataframe['close'])) / (dataframe['high'] - dataframe['low'] + 1e-10)
        mf_volume = mf_multiplier * dataframe['volume']
        dataframe['cmf'] = mf_volume.rolling(20).sum() / dataframe['volume'].rolling(20).sum().replace(0, 1e-10)

        # ── Donchian Channels ─────────────────────────────────────────────
        for lb in [20, 55, 100]:
            dataframe[f'dc_high_{lb}'] = dataframe['high'].shift(1).rolling(lb).max()
            dataframe[f'dc_low_{lb}'] = dataframe['low'].shift(1).rolling(lb).min()

        score = np.zeros(len(dataframe))
        for lb in [20, 55, 100]:
            sig = np.where(
                dataframe['close'] > dataframe[f'dc_high_{lb}'], 1.0,
                np.where(dataframe['close'] < dataframe[f'dc_low_{lb}'], -1.0, 0.0)
            )
            score += sig
        dataframe['trend_score'] = score / 3.0

        # ── Donchian midline for exit ─────────────────────────────────────
        dataframe['dc_mid_20'] = (dataframe['dc_high_20'] + dataframe['dc_low_20']) / 2

        # ── Candlestick Patterns ──────────────────────────────────────────
        body = (dataframe['close'] - dataframe['open']).abs()
        candle_range = dataframe['high'] - dataframe['low']
        upper_wick = dataframe['high'] - dataframe[['open', 'close']].max(axis=1)
        lower_wick = dataframe[['open', 'close']].min(axis=1) - dataframe['low']

        # Hammer
        dataframe['hammer'] = (
            (lower_wick > 2 * body) &
            (upper_wick < 0.3 * body) &
            (dataframe['close'] > dataframe['open'])
        ).astype(float)

        # Bullish Engulfing
        dataframe['engulfing'] = (
            (dataframe['close'].shift(1) < dataframe['open'].shift(1)) &
            (dataframe['close'] > dataframe['open']) &
            (dataframe['open'] < dataframe['close'].shift(1)) &
            (dataframe['close'] > dataframe['open'].shift(1))
        ).astype(float)

        # Morning Star
        dataframe['morning_star'] = (
            (dataframe['close'].shift(2) < dataframe['open'].shift(2)) &
            (body.shift(1) < body.shift(2) * 0.3) &
            (dataframe['close'] > dataframe['open']) &
            (dataframe['close'] > (dataframe['open'].shift(2) + dataframe['close'].shift(2)) / 2)
        ).astype(float)

        # Three White Soldiers
        dataframe['three_white_soldiers'] = (
            (dataframe['close'] > dataframe['open']) &
            (dataframe['close'].shift(1) > dataframe['open'].shift(1)) &
            (dataframe['close'].shift(2) > dataframe['open'].shift(2)) &
            (dataframe['close'] > dataframe['close'].shift(1)) &
            (dataframe['close'].shift(1) > dataframe['close'].shift(2)) &
            (dataframe['open'] > dataframe['open'].shift(1)) &
            (dataframe['open'].shift(1) > dataframe['open'].shift(2))
        ).astype(float)

        # Doji Reversal
        dataframe['doji'] = (body < candle_range * 0.1).astype(float)
        dataframe['doji_reversal'] = (
            (dataframe['doji'] == 1) &
            (dataframe['close'].shift(1) < dataframe['close'].shift(3))
        ).astype(float)

        # Piercing Line
        dataframe['piercing_line'] = (
            (dataframe['close'].shift(1) < dataframe['open'].shift(1)) &
            (dataframe['open'] < dataframe['low'].shift(1)) &
            (dataframe['close'] > (dataframe['open'].shift(1) + dataframe['close'].shift(1)) / 2) &
            (dataframe['close'] < dataframe['open'].shift(1))
        ).astype(float)

        # ── Realized Volatility ───────────────────────────────────────────
        dataframe['returns'] = dataframe['close'].pct_change()
        dataframe['realized_vol'] = dataframe['returns'].rolling(30).std() * np.sqrt(365 * 24)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Base condition - Donchian breakout
        conditions = [dataframe['trend_score'] >= self.dc_trend_score_entry.value]

        # Volume confirmation (always required)
        conditions.append(dataframe['volume_ratio'] >= self.buy_volume_factor.value)
        conditions.append(dataframe['atr'] > 0)
        conditions.append(dataframe['volume'] > 0)

        # ── NEW: ADX Regime Gate ──────────────────────────────────────────
        # Only enter when market is actually trending (ADX > threshold)
        if self.use_adx_regime.value:
            conditions.append(dataframe['adx'] >= self.adx_regime_min.value)

        # ── NEW: 200 EMA Trend Filter ─────────────────────────────────────
        # Only enter longs when price is above 200 EMA (with trend)
        if self.use_ema200_filter.value:
            conditions.append(dataframe['close'] > dataframe['ema_200'])

        # ── NEW: ATR Close Confirmation Buffer ────────────────────────────
        # Require close above Donchian high by at least N * ATR
        # Filters out wick-driven false breakouts
        if self.use_atr_confirm.value:
            conditions.append(
                dataframe['close'] > dataframe['dc_high_20'] + (self.atr_confirm_mult.value * dataframe['atr'])
            )

        # Optional indicators
        if self.use_rsi.value:
            conditions.append(dataframe['rsi'] < self.buy_rsi_max.value)
            conditions.append(dataframe['rsi'] > self.buy_rsi_min.value)

        if self.use_stoch_rsi.value:
            conditions.append(dataframe['stoch_rsi'] < self.buy_stoch_rsi_max.value)

        if self.use_macd.value:
            conditions.append(dataframe['macd'] > dataframe['macd_signal'])
            conditions.append(dataframe['macd_hist'] > 0)

        if self.use_ema_9_21.value:
            conditions.append(dataframe['ema_9'] > dataframe['ema_21'])

        if self.use_ema_21_50.value:
            conditions.append(dataframe['ema_21'] > dataframe['ema_50'])

        if self.use_ema_50_200.value:
            conditions.append(dataframe['ema_50'] > dataframe['ema_200'])

        if self.use_adx.value:
            conditions.append(dataframe['adx'] > self.buy_adx_min.value)
            conditions.append(dataframe['plus_di'] > dataframe['minus_di'])

        if self.use_bb_breakout.value:
            conditions.append(dataframe['close'] > dataframe['bb_upper'].shift(1))

        if self.use_cci.value:
            conditions.append(dataframe['cci'] > self.buy_cci_min.value)

        if self.use_williams.value:
            conditions.append(dataframe['williams_r'] > self.buy_williams_max.value)

        if self.use_roc.value:
            conditions.append(dataframe['roc'] > self.buy_roc_min.value)

        if self.use_cmf.value:
            conditions.append(dataframe['cmf'] > self.buy_cmf_min.value)

        if self.use_obv_trend.value:
            conditions.append(dataframe['obv_trend'] == 1)

        # Candlestick patterns
        pattern_conditions = []
        if self.use_hammer.value:
            pattern_conditions.append(dataframe['hammer'] == 1)
        if self.use_engulfing.value:
            pattern_conditions.append(dataframe['engulfing'] == 1)
        if self.use_morning_star.value:
            pattern_conditions.append(dataframe['morning_star'] == 1)
        if self.use_three_white_soldiers.value:
            pattern_conditions.append(dataframe['three_white_soldiers'] == 1)
        if self.use_doji_reversal.value:
            pattern_conditions.append(dataframe['doji_reversal'] == 1)
        if self.use_piercing_line.value:
            pattern_conditions.append(dataframe['piercing_line'] == 1)

        if pattern_conditions:
            import functools
            import operator
            combined = functools.reduce(operator.or_, pattern_conditions)
            conditions.append(combined)

        import functools
        import operator
        final_condition = functools.reduce(operator.and_, conditions)
        dataframe.loc[final_condition, 'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit signal disabled — using custom_exit with profit gate instead
        dataframe.loc[:, 'exit_long'] = 0
        return dataframe

    def custom_exit(self, pair: str, trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return None

        last = dataframe.iloc[-1]

        # ── Time-based stop: exit losing trades after 48 hours ────────────
        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 3600
        if current_profit < 0 and trade_duration > 48:
            return 'time_stop_48h'

        # ── Donchian midline re-entry exit for losing trades ──────────────
        if current_profit < -0.03 and last['close'] < last['dc_mid_20']:
            return 'dc_midline_exit'

        # ── Profit gate custom exit ───────────────────────────────────────
        if current_profit < self.exit_min_profit.value:
            return None

        # Donchian breakdown required
        if last['trend_score'] > self.dc_trend_score_exit.value:
            return None

        # Optional MACD confirmation
        if self.use_macd_exit.value:
            if last['macd'] >= last['macd_signal']:
                return None

        # Optional BB confirmation
        if self.use_bb_exit.value:
            if last['close'] >= last['bb_mid']:
                return None

        return 'custom_exit_signal'

    def custom_stoploss(self, pair: str, trade, current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return self.stoploss
        last_candle = dataframe.iloc[-1]
        atr = last_candle['atr']
        if atr <= 0 or current_rate <= 0:
            return self.stoploss
        return -(self.atr_multiplier.value * atr) / current_rate

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag, side: str,
                            **kwargs) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return False
        if dataframe.iloc[-1]['realized_vol'] > 5.0:
            return False
        return True
