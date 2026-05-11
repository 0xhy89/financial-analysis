from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

import pandas as pd
import numpy as np

from config import (
    REPORTS_DIR,
    LOG_LEVEL,
    LOG_FILE,
    MARKETS,
    ASHARE_POLICY_SECTORS,
    US_POLICY_SECTORS,
    HK_POLICY_SECTORS
)
from utils import (
    setup_logger,
    save_to_csv,
    save_to_json,
    format_percentage,
    format_market_cap,
    format_report_date
)
from stock_pool import StockInfo, StockPool, StockPoolManager
from data_api import DataAPI


logger = setup_logger("DailyRecommend", LOG_FILE, LOG_LEVEL)


class SignalType(Enum):
    MA_BULLISH = "MA多头排列"
    MACD_GOLDEN_CROSS = "MACD金叉"
    MACD_BULLISH = "MACD柱状图扩张"
    BOLLINGER_BREAKOUT = "布林带突破"
    RSI_OVERSOLD_RECOVERY = "RSI超卖反弹"
    KDJ_GOLDEN_CROSS = "KDJ低位金叉"
    VOLUME_CONFIRMATION = "量价配合"
    BREAKTHROUGH = "有效突破"
    SUPPORT_REBOUND = "支撑位反弹"
    TREND_REVERSAL = "趋势反转"


@dataclass
class TechnicalSignal:
    signal_type: SignalType
    strength: float
    description: str
    date: datetime

    def to_dict(self) -> Dict:
        return {
            "signal_type": self.signal_type.value,
            "strength": self.strength,
            "description": self.description,
            "date": self.date.strftime("%Y-%m-%d")
        }


@dataclass
class SecurityMargin:
    valuation_level: str
    position_level: str
    bubble_risk: str
    overall: str
    pe_ratio: float = 0.0
    pb_ratio: float = 0.0
    distance_from_high: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "valuation_level": self.valuation_level,
            "position_level": self.position_level,
            "bubble_risk": self.bubble_risk,
            "overall": self.overall,
            "pe_ratio": self.pe_ratio,
            "pb_ratio": self.pb_ratio,
            "distance_from_high": self.distance_from_high
        }


@dataclass
class RecommendedStock:
    code: str
    name: str
    market: str
    sector: str
    policy_match: float
    security_margin: SecurityMargin
    technical_signals: List[TechnicalSignal]
    overall_score: float
    entry_range: Tuple[float, float]
    current_price: float = 0.0
    recommended_date: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "code": self.code,
            "name": self.name,
            "market": self.market,
            "sector": self.sector,
            "policy_match": format_percentage(self.policy_match),
            "security_margin": self.security_margin.to_dict(),
            "technical_signals": [s.to_dict() for s in self.technical_signals],
            "overall_score": self.overall_score,
            "entry_range": {
                "low": self.entry_range[0],
                "high": self.entry_range[1]
            },
            "current_price": self.current_price,
            "recommended_date": self.recommended_date.strftime("%Y-%m-%d")
        }


@dataclass
class DailyRecommendation:
    date: datetime
    market: str
    recommendations: List[RecommendedStock] = field(default_factory=list)

    def get_top_stocks(self, top_n: int = 5) -> List[RecommendedStock]:
        sorted_stocks = sorted(self.recommendations, key=lambda x: x.overall_score, reverse=True)
        return sorted_stocks[:top_n]

    def to_dataframe(self) -> pd.DataFrame:
        if not self.recommendations:
            return pd.DataFrame()

        data = []
        for stock in self.recommendations:
            data.append({
                "代码": stock.code,
                "名称": stock.name,
                "市场": stock.market,
                "所属赛道": stock.sector,
                "政策贴合度": format_percentage(stock.policy_match),
                "安全边际": stock.security_margin.overall,
                "技术信号数量": len(stock.technical_signals),
                "综合评分": f"{stock.overall_score:.1f}",
                "当前价格": f"{stock.current_price:.2f}",
                "建议入场低价": f"{stock.entry_range[0]:.2f}",
                "建议入场高价": f"{stock.entry_range[1]:.2f}",
                "推荐日期": stock.recommended_date.strftime("%Y-%m-%d")
            })

        return pd.DataFrame(data)


class TechnicalSignalAnalyzer:
    def __init__(self, config=None):
        self.ma_periods = [5, 10, 20, 60, 120]
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.kdj_period = 9
        self.bb_period = 20
        self.bb_std = 2

    def analyze(self, df: pd.DataFrame) -> Tuple[List[TechnicalSignal], float]:
        if df.empty or len(df) < 60:
            return [], 0.0

        signals = []

        ma_signals = self._check_ma_signals(df)
        signals.extend(ma_signals)

        macd_signals = self._check_macd_signals(df)
        signals.extend(macd_signals)

        rsi_signals = self._check_rsi_signals(df)
        signals.extend(rsi_signals)

        kdj_signals = self._check_kdj_signals(df)
        signals.extend(kdj_signals)

        volume_signals = self._check_volume_signals(df)
        signals.extend(volume_signals)

        bb_signals = self._check_bollinger_signals(df)
        signals.extend(bb_signals)

        technical_score = self._calculate_technical_score(signals, df)

        return signals, technical_score

    def _check_ma_signals(self, df: pd.DataFrame) -> List[TechnicalSignal]:
        signals = []

        close = df["close"]

        ma_5 = close.rolling(window=5).mean()
        ma_10 = close.rolling(window=10).mean()
        ma_20 = close.rolling(window=20).mean()
        ma_60 = close.rolling(window=60).mean()

        current = close.iloc[-1]
        ma5_current = ma_5.iloc[-1]
        ma10_current = ma_10.iloc[-1]
        ma20_current = ma_20.iloc[-1]
        ma60_current = ma_60.iloc[-1]

        if ma5_current > ma10_current > ma20_current > ma60_current:
            signals.append(TechnicalSignal(
                signal_type=SignalType.MA_BULLISH,
                strength=85.0,
                description="MA5>MA10>MA20>MA60 多头排列，上升趋势健康",
                date=df.index[-1]
            ))

        return signals

    def _check_macd_signals(self, df: pd.DataFrame) -> List[TechnicalSignal]:
        signals = []

        close = df["close"]

        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd = 2 * (exp1 - exp2)
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal

        if len(histogram) >= 2:
            current_hist = histogram.iloc[-1]
            prev_hist = histogram.iloc[-2]

            if current_hist > 0 and prev_hist <= 0:
                signals.append(TechnicalSignal(
                    signal_type=SignalType.MACD_GOLDEN_CROSS,
                    strength=80.0,
                    description="MACD金叉形成，看涨信号",
                    date=df.index[-1]
                ))

            if current_hist > prev_hist > 0:
                signals.append(TechnicalSignal(
                    signal_type=SignalType.MACD_BULLISH,
                    strength=75.0,
                    description="MACD柱状图扩张，上涨动能增强",
                    date=df.index[-1]
                ))

        return signals

    def _check_rsi_signals(self, df: pd.DataFrame) -> List[TechnicalSignal]:
        signals = []

        close = df["close"]
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]

        if prev_rsi < self.rsi_oversold and current_rsi >= self.rsi_oversold:
            signals.append(TechnicalSignal(
                signal_type=SignalType.RSI_OVERSOLD_RECOVERY,
                strength=78.0,
                description=f"RSI从超卖区域回升至{current_rsi:.1f}，技术反弹信号",
                date=df.index[-1]
            ))

        return signals

    def _check_kdj_signals(self, df: pd.DataFrame) -> List[TechnicalSignal]:
        signals = []

        low_min = df["low"].rolling(window=9).min()
        high_max = df["high"].rolling(window=9).max()
        rsv = (df["close"] - low_min) / (high_max - low_min) * 100

        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * d

        k_current = k.iloc[-1]
        d_current = d.iloc[-1]
        k_prev = k.iloc[-2]
        d_prev = d.iloc[-2]

        if k_prev < d_prev and k_current > d_current and k_current < 60:
            signals.append(TechnicalSignal(
                signal_type=SignalType.KDJ_GOLDEN_CROSS,
                strength=77.0,
                description=f"KDJ低位金叉(K={k_current:.1f}, D={d_current:.1f})，低位启动信号",
                date=df.index[-1]
            ))

        return signals

    def _check_volume_signals(self, df: pd.DataFrame) -> List[TechnicalSignal]:
        signals = []

        if "volume" not in df.columns:
            return signals

        close = df["close"]
        volume = df["volume"]

        vol_ma5 = volume.rolling(window=5).mean()
        vol_ma20 = volume.rolling(window=20).mean()

        close_ma5 = close.rolling(window=5).mean()
        close_ma20 = close.rolling(window=20).mean()

        if len(close) >= 2:
            price_trend = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]
            vol_ratio = volume.iloc[-1] / vol_ma20.iloc[-1] if vol_ma20.iloc[-1] > 0 else 0

            if vol_ratio > 1.5 and price_trend > 0:
                signals.append(TechnicalSignal(
                    signal_type=SignalType.VOLUME_CONFIRMATION,
                    strength=72.0,
                    description=f"量价齐升(量比={vol_ratio:.1f})，上涨得到量能确认",
                    date=df.index[-1]
                ))

        return signals

    def _check_bollinger_signals(self, df: pd.DataFrame) -> List[TechnicalSignal]:
        signals = []

        close = df["close"]
        ma = close.rolling(window=20).mean()
        std = close.rolling(window=20).std()
        upper_band = ma + 2 * std
        lower_band = ma - 2 * std

        current_close = close.iloc[-1]
        prev_close = close.iloc[-2]

        if prev_close < upper_band.iloc[-2] and current_close > upper_band.iloc[-1]:
            signals.append(TechnicalSignal(
                signal_type=SignalType.BOLLINGER_BREAKOUT,
                strength=73.0,
                description="股价突破布林带上轨，动能强劲",
                date=df.index[-1]
            ))

        return signals

    def _calculate_technical_score(self, signals: List[TechnicalSignal], df: pd.DataFrame) -> float:
        if not signals:
            return 50.0

        signal_score = 0.0
        for signal in signals:
            signal_score += signal.strength

        avg_signal_score = signal_score / len(signals)

        signal_count_bonus = min(len(signals) * 3, 15)

        trend_score = 50.0
        if len(df) >= 20:
            ma20 = df["close"].rolling(window=20).mean()
            if df["close"].iloc[-1] > ma20.iloc[-1]:
                trend_score += 15
            else:
                trend_score -= 15

        total_score = avg_signal_score * 0.7 + trend_score * 0.3 + signal_count_bonus

        return min(max(total_score, 0), 100)


class SecurityMarginAnalyzer:
    def __init__(self):
        self.high_distance_threshold = 0.70

    def analyze(
        self,
        df: pd.DataFrame,
        financial_data: Dict
    ) -> SecurityMargin:
        pe = financial_data.get("pe", 0)
        pb = financial_data.get("pb", 0)

        valuation_level = self._analyze_valuation(pe, pb)

        position_level = self._analyze_position(df)

        bubble_risk = self._analyze_bubble_risk(df)

        distance_from_high = self._calculate_distance_from_high(df)

        pe_ratio = pe
        pb_ratio = pb

        overall = self._calculate_overall(
            valuation_level,
            position_level,
            bubble_risk
        )

        return SecurityMargin(
            valuation_level=valuation_level,
            position_level=position_level,
            bubble_risk=bubble_risk,
            overall=overall,
            pe_ratio=pe_ratio,
            pb_ratio=pb_ratio,
            distance_from_high=distance_from_high
        )

    def _analyze_valuation(self, pe: float, pb: float) -> str:
        if pe <= 0 and pb <= 0:
            return "无法评估"

        pe_score = 0
        if 0 < pe <= 15:
            pe_score = 3
        elif 15 < pe <= 25:
            pe_score = 2
        elif 25 < pe <= 40:
            pe_score = 1
        elif pe > 40:
            pe_score = 0

        pb_score = 0
        if 0 < pb <= 2:
            pb_score = 3
        elif 2 < pb <= 4:
            pb_score = 2
        elif 4 < pb <= 8:
            pb_score = 1
        elif pb > 8:
            pb_score = 0

        total_score = pe_score + pb_score

        if total_score >= 5:
            return "低估值"
        elif total_score >= 3:
            return "合理估值"
        elif total_score >= 1:
            return "偏高估值"
        else:
            return "高估值"

    def _analyze_position(self, df: pd.DataFrame) -> str:
        if df.empty or len(df) < 60:
            return "无法评估"

        current_price = df["close"].iloc[-1]

        high_52w = df["high"].rolling(window=252).max().iloc[-1]
        low_52w = df["low"].rolling(window=252).min().iloc[-1]

        if high_52w == 0:
            return "无法评估"

        position_ratio = (current_price - low_52w) / (high_52w - low_52w)

        if position_ratio <= 0.30:
            return "低位"
        elif position_ratio <= 0.50:
            return "中低位"
        elif position_ratio <= 0.70:
            return "中间位"
        elif position_ratio <= 0.85:
            return "中高位置"
        else:
            return "高位"

    def _analyze_bubble_risk(self, df: pd.DataFrame) -> str:
        if df.empty or len(df) < 60:
            return "无法评估"

        current_price = df["close"].iloc[-1]

        ma60 = df["close"].rolling(window=60).mean().iloc[-1]
        ma120 = df["close"].rolling(window=120).mean().iloc[-1]

        if ma60 == 0 or ma120 == 0:
            return "无法评估"

        deviation = (current_price - ma60) / ma60

        if deviation > 0.50:
            return "高泡沫风险"
        elif deviation > 0.30:
            return "存在泡沫"
        elif deviation > 0.15:
            return "轻微泡沫"
        elif deviation > -0.15:
            return "正常范围"
        else:
            return "价值洼地"

    def _calculate_distance_from_high(self, df: pd.DataFrame) -> float:
        if df.empty or len(df) < 60:
            return 0.0

        current_price = df["close"].iloc[-1]
        high_52w = df["high"].rolling(window=252).max().iloc[-1]

        if high_52w == 0:
            return 0.0

        return (high_52w - current_price) / high_52w

    def _calculate_overall(
        self,
        valuation: str,
        position: str,
        bubble: str
    ) -> str:
        safe_count = 0

        if valuation in ["低估值", "合理估值"]:
            safe_count += 2
        elif valuation == "偏高估值":
            safe_count += 1

        if position in ["低位", "中低位"]:
            safe_count += 2
        elif position == "中间位":
            safe_count += 1

        if bubble in ["价值洼地", "正常范围", "轻微泡沫"]:
            safe_count += 2
        elif bubble == "存在泡沫":
            safe_count += 1

        if safe_count >= 5:
            return "高安全边际"
        elif safe_count >= 3:
            return "中等安全边际"
        elif safe_count >= 1:
            return "低安全边际"
        else:
            return "风险较高"


class EntryRangeCalculator:
    def calculate(
        self,
        df: pd.DataFrame,
        current_price: float,
        security_margin: SecurityMargin
    ) -> Tuple[float, float]:
        if df.empty or len(df) < 20:
            return current_price * 0.95, current_price * 1.05

        ma20 = df["close"].rolling(window=20).mean().iloc[-1]
        ma60 = df["close"].rolling(window=60).mean().iloc[-1]

        if security_margin.position_level in ["低位", "中低位", "中间位"]:
            support = min(ma20, ma60) * 0.98
            resistance = current_price * 1.05
        else:
            support = current_price * 0.95
            resistance = current_price * 1.10

        low_entry = (support + current_price) / 2 * 0.99
        high_entry = (resistance + current_price) / 2 * 1.01

        return round(low_entry, 2), round(high_entry, 2)


class DailyRecommendEngine:
    def __init__(self, data_api: Optional[DataAPI] = None):
        self.data_api = data_api or DataAPI()
        self.signal_analyzer = TechnicalSignalAnalyzer()
        self.margin_analyzer = SecurityMarginAnalyzer()
        self.entry_calculator = EntryRangeCalculator()

    def generate_recommendations(
        self,
        stock_pool: StockPool,
        market: str,
        top_n: int = 5
    ) -> DailyRecommendation:
        recommendation = DailyRecommendation(
            date=datetime.now(),
            market=market
        )

        market_configs = {
            "A股": {"count": 5, "sectors": ASHARE_POLICY_SECTORS},
            "美股": {"count": 3, "sectors": US_POLICY_SECTORS},
            "港股": {"count": 2, "sectors": HK_POLICY_SECTORS}
        }

        config = market_configs.get(market, {"count": 3, "sectors": []})
        target_count = min(top_n, config["count"])

        candidate_stocks = []

        for stock in stock_pool.stocks:
            if stock.policy_match < 0.3:
                continue

            try:
                df = self.data_api.get_daily_data(stock.code, market, "3mo")

                if df.empty or len(df) < 30:
                    continue

                technical_signals, technical_score = self.signal_analyzer.analyze(df)

                if not technical_signals:
                    continue

                financial_data = self.data_api.get_financial_data(stock.code, market)

                security_margin = self.margin_analyzer.analyze(df, financial_data)

                if security_margin.overall not in ["高安全边际", "中等安全边际"]:
                    continue

                current_price = df["close"].iloc[-1]
                entry_range = self.entry_calculator.calculate(
                    df, current_price, security_margin
                )

                final_score = (
                    stock.policy_match * 25 +
                    stock.financial_score * 25 +
                    technical_score * 30 +
                    (100 if security_margin.overall == "高安全边际" else 50) * 0.20
                )

                recommended_stock = RecommendedStock(
                    code=stock.code,
                    name=stock.name,
                    market=market,
                    sector=stock.sector,
                    policy_match=stock.policy_match,
                    security_margin=security_margin,
                    technical_signals=technical_signals,
                    overall_score=final_score,
                    entry_range=entry_range,
                    current_price=current_price
                )

                candidate_stocks.append(recommended_stock)

            except Exception as e:
                logger.error(f"Failed to analyze {stock.code}: {e}")
                continue

        candidate_stocks.sort(key=lambda x: x.overall_score, reverse=True)

        recommendation.recommendations = candidate_stocks[:target_count]

        logger.info(f"Generated {len(recommendation.recommendations)} recommendations for {market}")

        return recommendation

    def generate_all_market_recommendations(
        self,
        pool_manager: StockPoolManager,
        market_configs: Optional[Dict[str, int]] = None
    ) -> Dict[str, DailyRecommendation]:
        if market_configs is None:
            market_configs = {
                "A股": 5,
                "美股": 3,
                "港股": 2
            }

        all_recommendations = {}

        for market, top_n in market_configs.items():
            pool = pool_manager.get_pool(market)
            if pool:
                rec = self.generate_recommendations(pool, market, top_n)
                all_recommendations[market] = rec

        return all_recommendations


class ReportGenerator:
    def __init__(self, output_dir: Path = REPORTS_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def generate_text_report(
        self,
        recommendations: Dict[str, DailyRecommendation]
    ) -> str:
        lines = []
        lines.append("=" * 100)
        lines.append("全市场智能量化选股 - 每日推荐报告")
        lines.append(f"报告生成时间: {format_report_date()}")
        lines.append("=" * 100)

        for market, rec in recommendations.items():
            lines.append(f"\n{'=' * 60}")
            lines.append(f"市场: {market}")
            lines.append(f"推荐标的数量: {len(rec.recommendations)}")
            lines.append("=" * 60)

            if not rec.recommendations:
                lines.append("暂无推荐标的")
                continue

            lines.append("\n" + "-" * 100)
            lines.append(f"{'代码':<10} {'名称':<12} {'所属赛道':<15} {'政策匹配':<10} {'安全边际':<12} "
                        f"{'技术信号':<8} {'综合评分':<10} {'当前价格':<10} {'入场区间':<25}")
            lines.append("-" * 100)

            for stock in rec.recommendations:
                signal_count = len(stock.technical_signals)
                entry_range = f"{stock.entry_range[0]:.2f}-{stock.entry_range[1]:.2f}"

                lines.append(
                    f"{stock.code:<10} {stock.name:<12} {stock.sector:<15} "
                    f"{stock.policy_match:.1%}   "
                    f"{stock.security_margin.overall:<12} "
                    f"{signal_count:<8} "
                    f"{stock.overall_score:<10.1f} "
                    f"{stock.current_price:<10.2f} "
                    f"{entry_range:<25}"
                )

            for stock in rec.recommendations:
                lines.append(f"\n{'─' * 60}")
                lines.append(f"标的: {stock.name}({stock.code}) - {stock.market}")
                lines.append(f"所属赛道: {stock.sector}")
                lines.append(f"政策贴合度: {stock.policy_match:.1%}")
                lines.append(f"安全边际说明: {stock.security_margin.overall}")
                lines.append(f"  - 估值水平: {stock.security_margin.valuation_level} (PE={stock.security_margin.pe_ratio:.1f}, PB={stock.security_margin.pb_ratio:.1f})")
                lines.append(f"  - 位置水平: {stock.security_margin.position_level} (距高点{stock.security_margin.distance_from_high:.1%})")
                lines.append(f"  - 泡沫风险: {stock.security_margin.bubble_risk}")

                lines.append(f"\n技术面触发信号 ({len(stock.technical_signals)}个):")
                for i, signal in enumerate(stock.technical_signals, 1):
                    lines.append(f"  {i}. [{signal.signal_type.value}] {signal.description} (强度:{signal.strength:.0f})")

                lines.append(f"\n建议观察/入场区间: {stock.entry_range[0]:.2f} - {stock.entry_range[1]:.2f}")
                lines.append(f"当前价格: {stock.current_price:.2f}")

        lines.append("\n" + "=" * 100)
        lines.append("免责声明: 本报告仅供参考，不构成投资建议。投资有风险，入市需谨慎。")
        lines.append("=" * 100)

        return "\n".join(lines)

    def save_report(
        self,
        recommendations: Dict[str, DailyRecommendation],
        format: str = "txt"
    ) -> Path:
        if format == "txt":
            return self._save_text_report(recommendations)
        elif format == "csv":
            return self._save_csv_report(recommendations)
        elif format == "json":
            return self._save_json_report(recommendations)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _save_text_report(self, recommendations: Dict[str, DailyRecommendation]) -> Path:
        report_content = self.generate_text_report(recommendations)

        filename = f"daily_recommend_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"Saved text report to {filepath}")
        return filepath

    def _save_csv_report(self, recommendations: Dict[str, DailyRecommendation]) -> Path:
        all_data = []

        for market, rec in recommendations.items():
            for stock in rec.recommendations:
                all_data.append(stock.to_dict())

        if all_data:
            df = pd.DataFrame(all_data)
            filename = f"daily_recommend_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            filepath = self.output_dir / filename
            save_to_csv(df, filepath)
            logger.info(f"Saved CSV report to {filepath}")
            return filepath

        return self.output_dir / "empty_report.csv"

    def _save_json_report(self, recommendations: Dict[str, DailyRecommendation]) -> Path:
        data = {}
        for market, rec in recommendations.items():
            data[market] = {
                "date": rec.date.strftime("%Y-%m-%d"),
                "recommendations": [s.to_dict() for s in rec.recommendations]
            }

        filename = f"daily_recommend_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        filepath = self.output_dir / filename
        save_to_json(data, filepath)
        logger.info(f"Saved JSON report to {filepath}")
        return filepath


class DailyRecommendSystem:
    def __init__(self):
        self.data_api = DataAPI()
        self.recommend_engine = DailyRecommendEngine(self.data_api)
        self.report_generator = ReportGenerator()
        self.pool_manager = StockPoolManager()

    def run(
        self,
        stock_pools: Dict[str, StockPool],
        market_configs: Optional[Dict[str, int]] = None,
        save_formats: List[str] = None
    ) -> Dict[str, DailyRecommendation]:
        for market, pool in stock_pools.items():
            self.pool_manager.add_pool(pool)

        recommendations = self.recommend_engine.generate_all_market_recommendations(
            self.pool_manager,
            market_configs
        )

        if save_formats is None:
            save_formats = ["txt", "csv", "json"]

        for fmt in save_formats:
            try:
                self.report_generator.save_report(recommendations, fmt)
            except Exception as e:
                logger.error(f"Failed to save {fmt} report: {e}")

        self._print_summary(recommendations)

        return recommendations

    def _print_summary(self, recommendations: Dict[str, DailyRecommendation]) -> None:
        print("\n" + "=" * 80)
        print("每日推荐完成!")
        print("=" * 80)

        for market, rec in recommendations.items():
            print(f"\n{market}: {len(rec.recommendations)}只推荐标的")
            for stock in rec.get_top_stocks(3):
                print(f"  - {stock.code} {stock.name} (评分:{stock.overall_score:.1f})")


if __name__ == "__main__":
    from stock_pool import StockInfo, StockPool

    mock_pool = StockPool(name="测试", market="A股")

    stocks = [
        StockInfo(
            code="600519",
            name="贵州茅台",
            market="A股",
            sector="白酒",
            policy_match=0.8,
            pe=35.0,
            pb=12.0,
            roe=25.0,
            market_cap=250_000_000_000,
            financial_score=75.0,
            technical_score=80.0,
            overall_score=78.0
        ),
        StockInfo(
            code="002475",
            name="立讯精密",
            market="A股",
            sector="消费电子",
            policy_match=0.7,
            pe=25.0,
            pb=5.0,
            roe=18.0,
            market_cap=180_000_000_000,
            financial_score=70.0,
            technical_score=85.0,
            overall_score=75.0
        )
    ]

    for stock in stocks:
        mock_pool.add_stock(stock)

    system = DailyRecommendSystem()

    recommendations = system.run(
        {"A股": mock_pool},
        {"A股": 3}
    )

    print("\n推荐报告:")
    report = system.report_generator.generate_text_report(recommendations)
    print(report)
