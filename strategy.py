from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import pandas as pd
import numpy as np

from config import (
    TECHNICAL_CONFIG,
    LOG_LEVEL,
    LOG_FILE
)
from utils import (
    setup_logger,
    calculate_returns,
    calculate_cumulative_returns
)


logger = setup_logger("Strategy", LOG_FILE, LOG_LEVEL)


class StrategyType(Enum):
    TREND_FOLLOWING = "趋势跟踪"
    MEAN_REVERSION = "均值回归"
    BREAKOUT = "突破策略"
    MOMENTUM = "动量策略"
    COMBINED = "综合策略"


@dataclass
class TradeSignal:
    date: datetime
    symbol: str
    signal_type: str
    price: float
    strength: float
    reason: str


@dataclass
class StrategyResult:
    strategy_name: str
    strategy_type: StrategyType
    signals: List[TradeSignal] = field(default_factory=list)
    performance_metrics: Dict = field(default_factory=dict)

    def get_buy_signals(self) -> List[TradeSignal]:
        return [s for s in self.signals if s.signal_type == "BUY"]

    def get_sell_signals(self) -> List[TradeSignal]:
        return [s for s in self.signals if s.signal_type == "SELL"]


class TechnicalIndicators:
    @staticmethod
    def calculate_ma(prices: pd.Series, period: int) -> pd.Series:
        return prices.rolling(window=period).mean()

    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
        return prices.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_macd(
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = 2 * (exp1 - exp2)
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

    @staticmethod
    def calculate_bollinger_bands(
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)
        return upper_band, ma, lower_band

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_kdj(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 9,
        smoothing: int = 3
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        rsv = (close - lowest_low) / (highest_high - lowest_low) * 100

        k = rsv.ewm(com=smoothing - 1, adjust=False).mean()
        d = k.ewm(com=smoothing - 1, adjust=False).mean()
        j = 3 * k - 2 * d
        return k, d, j

    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv

    @staticmethod
    def calculate_atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr

    @staticmethod
    def calculate_volume_ma(volume: pd.Series, period: int = 20) -> pd.Series:
        return volume.rolling(window=period).mean()

    @staticmethod
    def calculate_rps(close: pd.Series, period: int = 252) -> pd.Series:
        returns = close.pct_change()
        rps = (returns.rolling(window=period).mean() / returns.rolling(window=period).std()) * 100
        return rps.fillna(0)


class TrendFollowingStrategy:
    def __init__(self, config: TECHNICAL_CONFIG = TECHNICAL_CONFIG):
        self.config = config
        self.indicators = TechnicalIndicators()

    def analyze(self, df: pd.DataFrame, symbol: str) -> StrategyResult:
        result = StrategyResult(
            strategy_name="趋势跟踪策略",
            strategy_type=StrategyType.TREND_FOLLOWING
        )

        if df.empty or len(df) < 60:
            return result

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        ma5 = self.indicators.calculate_ma(close, 5)
        ma10 = self.indicators.calculate_ma(close, 10)
        ma20 = self.indicators.calculate_ma(close, 20)
        ma60 = self.indicators.calculate_ma(close, 60)

        macd, signal_line, histogram = self.indicators.calculate_macd(
            close,
            self.config.macd_fast,
            self.config.macd_slow,
            self.config.macd_signal
        )

        for i in range(60, len(close)):
            current_date = close.index[i]
            current_price = close.iloc[i]

            prev_ma5 = ma5.iloc[i - 1]
            curr_ma5 = ma5.iloc[i]
            prev_ma10 = ma10.iloc[i - 1]
            curr_ma10 = ma10.iloc[i]
            prev_ma20 = ma20.iloc[i - 1]
            curr_ma20 = ma20.iloc[i]

            if (prev_ma5 < prev_ma10 < prev_ma20 and
                curr_ma5 > curr_ma10 > curr_ma20):
                signal = TradeSignal(
                    date=current_date,
                    symbol=symbol,
                    signal_type="BUY",
                    price=current_price,
                    strength=80.0,
                    reason="MA多头排列形成，短期、中期、长期均线呈多头态势"
                )
                result.signals.append(signal)

            elif (prev_ma5 > prev_ma10 > prev_ma20 and
                  curr_ma5 < curr_ma10 < curr_ma20):
                signal = TradeSignal(
                    date=current_date,
                    symbol=symbol,
                    signal_type="SELL",
                    price=current_price,
                    strength=80.0,
                    reason="MA空头排列形成，趋势转空"
                )
                result.signals.append(signal)

            prev_hist = histogram.iloc[i - 1]
            curr_hist = histogram.iloc[i]
            if prev_hist < 0 and curr_hist > 0:
                signal = TradeSignal(
                    date=current_date,
                    symbol=symbol,
                    signal_type="BUY",
                    price=current_price,
                    strength=75.0,
                    reason="MACD柱状图由负转正，上涨动能增强"
                )
                result.signals.append(signal)

        return result


class MeanReversionStrategy:
    def __init__(self, config: TECHNICAL_CONFIG = TECHNICAL_CONFIG):
        self.config = config
        self.indicators = TechnicalIndicators()

    def analyze(self, df: pd.DataFrame, symbol: str) -> StrategyResult:
        result = StrategyResult(
            strategy_name="均值回归策略",
            strategy_type=StrategyType.MEAN_REVERSION
        )

        if df.empty or len(df) < 60:
            return result

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        bb_upper, bb_middle, bb_lower = self.indicators.calculate_bollinger_bands(
            close,
            self.config.bollinger_period,
            self.config.bollinger_std
        )

        rsi = self.indicators.calculate_rsi(close, self.config.rsi_period)

        k, d, j = self.indicators.calculate_kdj(
            high, low, close,
            self.config.kdj_period,
            self.config.kdj_smoothing
        )

        for i in range(60, len(close)):
            current_date = close.index[i]
            current_price = close.iloc[i]
            current_rsi = rsi.iloc[i]
            current_k = k.iloc[i]
            current_d = d.iloc[i]
            prev_k = k.iloc[i - 1]
            prev_d = d.iloc[i - 1]

            if (current_price < bb_lower.iloc[i] and
                current_rsi < self.config.rsi_oversold):
                signal = TradeSignal(
                    date=current_date,
                    symbol=symbol,
                    signal_type="BUY",
                    price=current_price,
                    strength=85.0,
                    reason=f"价格触及布林带下轨且RSI超卖({current_rsi:.1f})，反弹概率高"
                )
                result.signals.append(signal)

            if (prev_k < prev_d and current_k > current_d and
                current_k < 40 and current_rsi < 50):
                signal = TradeSignal(
                    date=current_date,
                    symbol=symbol,
                    signal_type="BUY",
                    price=current_price,
                    strength=78.0,
                    reason=f"KDJ低位金叉(K={current_k:.1f}, D={current_d:.1f})，技术面转强"
                )
                result.signals.append(signal)

            if current_price > bb_upper.iloc[i] and current_rsi > self.config.rsi_overbought:
                signal = TradeSignal(
                    date=current_date,
                    symbol=symbol,
                    signal_type="SELL",
                    price=current_price,
                    strength=75.0,
                    reason=f"价格触及布林带上轨且RSI超买({current_rsi:.1f})，回调风险大"
                )
                result.signals.append(signal)

        return result


class BreakoutStrategy:
    def __init__(self, config: TECHNICAL_CONFIG = TECHNICAL_CONFIG):
        self.config = config
        self.indicators = TechnicalIndicators()

    def analyze(self, df: pd.DataFrame, symbol: str) -> StrategyResult:
        result = StrategyResult(
            strategy_name="突破策略",
            strategy_type=StrategyType.BREAKOUT
        )

        if df.empty or len(df) < 60:
            return result

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        ma20 = self.indicators.calculate_ma(close, 20)
        ma60 = self.indicators.calculate_ma(close, 60)

        vol_ma20 = self.indicators.calculate_volume_ma(volume, 20)

        for i in range(60, len(close)):
            current_date = close.index[i]
            current_price = close.iloc[i]
            current_volume = volume.iloc[i]
            prev_volume = volume.iloc[i - 1]

            high_20 = high.iloc[i - 20:i].max()
            low_20 = low.iloc[i - 20:i].min()

            volume_ratio = current_volume / vol_ma20.iloc[i] if vol_ma20.iloc[i] > 0 else 0

            if (current_price > high_20 and
                volume_ratio > self.config.min_breakthrough_volume_ratio):
                signal = TradeSignal(
                    date=current_date,
                    symbol=symbol,
                    signal_type="BUY",
                    price=current_price,
                    strength=82.0,
                    reason=f"股价突破20日高点({high_20:.2f})，量比{volume_ratio:.1f}，突破有效"
                )
                result.signals.append(signal)

            if (current_price < low_20 and
                volume_ratio > self.config.min_breakthrough_volume_ratio):
                signal = TradeSignal(
                    date=current_date,
                    symbol=symbol,
                    signal_type="SELL",
                    price=current_price,
                    strength=80.0,
                    reason=f"股价跌破20日低点({low_20:.2f})，下行趋势确立"
                )
                result.signals.append(signal)

        return result


class MomentumStrategy:
    def __init__(self, config: TECHNICAL_CONFIG = TECHNICAL_CONFIG):
        self.config = config
        self.indicators = TechnicalIndicators()

    def analyze(self, df: pd.DataFrame, symbol: str) -> StrategyResult:
        result = StrategyResult(
            strategy_name="动量策略",
            strategy_type=StrategyType.MOMENTUM
        )

        if df.empty or len(df) < 252:
            return result

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        rps = self.indicators.calculate_rps(close, 252)

        rsi = self.indicators.calculate_rsi(close, self.config.rsi_period)

        for i in range(252, len(close)):
            current_date = close.index[i]
            current_price = close.iloc[i]
            current_rps = rps.iloc[i]
            current_rsi = rsi.iloc[i]

            if current_rps > self.config.min_rps and current_rsi > 50:
                signal = TradeSignal(
                    date=current_date,
                    symbol=symbol,
                    signal_type="BUY",
                    price=current_price,
                    strength=80.0,
                    reason=f"RPS强度{current_rps:.1f}超过阈值，RSI={current_rsi:.1f}处于强势区域"
                )
                result.signals.append(signal)

            elif current_rps < -self.config.min_rps and current_rsi < 50:
                signal = TradeSignal(
                    date=current_date,
                    symbol=symbol,
                    signal_type="SELL",
                    price=current_price,
                    strength=80.0,
                    reason=f"RPS强度{current_rps:.1f}低于负阈值，股价走弱"
                )
                result.signals.append(signal)

        return result


class CombinedStrategy:
    def __init__(self, config: TECHNICAL_CONFIG = TECHNICAL_CONFIG):
        self.config = config
        self.trend_strategy = TrendFollowingStrategy(config)
        self.reversion_strategy = MeanReversionStrategy(config)
        self.breakout_strategy = BreakoutStrategy(config)
        self.momentum_strategy = MomentumStrategy(config)

    def analyze(self, df: pd.DataFrame, symbol: str) -> StrategyResult:
        result = StrategyResult(
            strategy_name="综合策略",
            strategy_type=StrategyType.COMBINED
        )

        trend_result = self.trend_strategy.analyze(df, symbol)
        reversion_result = self.reversion_strategy.analyze(df, symbol)
        breakout_result = self.breakout_strategy.analyze(df, symbol)
        momentum_result = self.momentum_strategy.analyze(df, symbol)

        all_signals = []
        all_signals.extend(trend_result.signals)
        all_signals.extend(reversion_result.signals)
        all_signals.extend(breakout_result.signals)
        all_signals.extend(momentum_result.signals)

        all_signals.sort(key=lambda x: (x.date, -x.strength))

        result.signals = all_signals

        return result

    def get_consensus_signals(
        self,
        df: pd.DataFrame,
        symbol: str,
        min_agreement: int = 2
    ) -> List[TradeSignal]:
        trend_result = self.trend_strategy.analyze(df, symbol)
        reversion_result = self.reversion_strategy.analyze(df, symbol)
        breakout_result = self.breakout_strategy.analyze(df, symbol)

        consensus_signals = []

        signal_groups = {}
        for signal in trend_result.signals + reversion_result.signals + breakout_result.signals:
            key = (signal.date, signal.signal_type)
            if key not in signal_groups:
                signal_groups[key] = []
            signal_groups[key].append(signal)

        for key, signals in signal_groups.items():
            if len(signals) >= min_agreement:
                avg_strength = sum(s.signal_type for s in signals) / len(signals)
                reasons = [s.reason for s in signals]
                combined_reason = " | ".join(set(reasons[:3]))

                consensus_signals.append(TradeSignal(
                    date=key[0],
                    symbol=symbol,
                    signal_type=key[1],
                    price=signals[0].price,
                    strength=avg_strength,
                    reason=combined_reason
                ))

        return consensus_signals


class StrategySelector:
    def __init__(self):
        self.strategies = {
            StrategyType.TREND_FOLLOWING: TrendFollowingStrategy(),
            StrategyType.MEAN_REVERSION: MeanReversionStrategy(),
            StrategyType.BREAKOUT: BreakoutStrategy(),
            StrategyType.MOMENTUM: MomentumStrategy(),
            StrategyType.COMBINED: CombinedStrategy(),
        }

    def select_strategy(self, strategy_type: StrategyType) -> any:
        return self.strategies.get(strategy_type)

    def analyze_with_strategy(
        self,
        df: pd.DataFrame,
        symbol: str,
        strategy_type: StrategyType
    ) -> StrategyResult:
        strategy = self.select_strategy(strategy_type)
        if strategy is None:
            return StrategyResult(
                strategy_name="Unknown",
                strategy_type=strategy_type
            )
        return strategy.analyze(df, symbol)

    def analyze_all_strategies(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> Dict[StrategyType, StrategyResult]:
        results = {}
        for strategy_type, strategy in self.strategies.items():
            results[strategy_type] = strategy.analyze(df, symbol)
        return results


class StrategyEngine:
    def __init__(self):
        self.selector = StrategySelector()
        self.strategy_scores: Dict[str, float] = {}

    def score_stock(
        self,
        df: pd.DataFrame,
        symbol: str,
        strategy_type: StrategyType = StrategyType.COMBINED
    ) -> float:
        result = self.selector.analyze_with_strategy(df, symbol, strategy_type)

        if not result.signals:
            return 50.0

        buy_signals = result.get_buy_signals()
        sell_signals = result.get_sell_signals()

        buy_score = sum(s.strength for s in buy_signals) / max(len(buy_signals), 1)
        sell_score = sum(s.strength for s in sell_signals) / max(len(sell_signals), 1)

        net_score = buy_score - sell_score * 0.5

        time_weight = 1.0
        if buy_signals:
            last_buy = max(s.date for s in buy_signals)
            days_since_buy = (df.index[-1] - last_buy).days
            if days_since_buy <= 5:
                time_weight = 1.2
            elif days_since_buy <= 20:
                time_weight = 1.0
            else:
                time_weight = 0.8

        final_score = min(max(net_score * time_weight, 0), 100)

        self.strategy_scores[symbol] = final_score

        return final_score

    def rank_stocks(
        self,
        stock_data: Dict[str, pd.DataFrame],
        strategy_type: StrategyType = StrategyType.COMBINED
    ) -> List[Tuple[str, float]]:
        scores = []
        for symbol, df in stock_data.items():
            score = self.score_stock(df, symbol, strategy_type)
            scores.append((symbol, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        return scores

    def get_top_signals(
        self,
        df: pd.DataFrame,
        symbol: str,
        strategy_type: StrategyType = StrategyType.COMBINED,
        top_n: int = 5
    ) -> List[TradeSignal]:
        result = self.selector.analyze_with_strategy(df, symbol, strategy_type)

        buy_signals = result.get_buy_signals()

        buy_signals.sort(key=lambda x: x.strength, reverse=True)

        return buy_signals[:top_n]


if __name__ == "__main__":
    import yfinance as yf

    print("Testing Strategy Module...")

    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="6mo")

    if not df.empty:
        engine = StrategyEngine()

        score = engine.score_stock(df, "AAPL", StrategyType.COMBINED)
        print(f"AAPL Combined Strategy Score: {score:.2f}")

        signals = engine.get_top_signals(df, "AAPL", StrategyType.COMBINED)
        print(f"\nTop Buy Signals for AAPL:")
        for signal in signals[:5]:
            print(f"  [{signal.date.strftime('%Y-%m-%d')}] {signal.reason} (Strength: {signal.strength:.0f})")
