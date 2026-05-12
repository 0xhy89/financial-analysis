from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List
import os


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"

DATA_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


@dataclass
class MarketConfig:
    name: str
    enabled: bool = True
    data_update_time: str = "08:30"
    recommend_count: int = 5


@dataclass
class AShareConfig(MarketConfig):
    name: str = "A股"
    recommend_count: int = 5


@dataclass
class HKConfig(MarketConfig):
    name: str = "港股"
    recommend_count: int = 2


@dataclass
class USConfig(MarketConfig):
    name: str = "美股"
    recommend_count: int = 3


@dataclass
class DataConfig:
    cache_enabled: bool = True
    cache_days: int = 1
    retry_times: int = 3
    retry_delay: float = 2.0
    timeout: int = 30


@dataclass
class FilterConfig:
    min_market_cap: float = 50_000_000_000
    max_pe: float = 50.0
    min_pe: float = 0.0
    max_pb: float = 10.0
    min_roe: float = 5.0
    max_shares_pledge: float = 0.30
    max_goodwill_ratio: float = 0.20
    exclude_st: bool = True
    exclude_high_position: bool = True
    position_ratio_threshold: float = 0.80


@dataclass
class TechnicalConfig:
    ma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 60, 120])
    ema_periods: List[int] = field(default_factory=lambda: [12, 26, 50, 200])
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    kdj_period: int = 9
    kdj_smoothing: int = 3
    volume_ma_periods: List[int] = field(default_factory=lambda: [5, 10, 20])
    min_rps: float = 70.0
    min_breakthrough_volume_ratio: float = 1.5


@dataclass
class BacktestConfig:
    initial_cash: float = 1_000_000.0
    max_position: float = 0.20
    max_total_position: float = 0.80
    stop_loss: float = 0.07
    take_profit: float = 0.20
    max_drawdown: float = 0.15
    commission: float = 0.0003
    slippage: float = 0.0001


@dataclass
class ReportConfig:
    output_format: List[str] = field(default_factory=lambda: ["txt", "csv"])
    include_charts: bool = True
    chart_format: str = "png"
    daily_recommend_filename: str = "daily_recommend"


MARKETS: Dict[str, MarketConfig] = {
    "A股": AShareConfig(name="A股", recommend_count=5),
    "港股": HKConfig(name="港股", recommend_count=2),
    "美股": USConfig(name="美股", recommend_count=3),
}

DATA_CONFIG = DataConfig()
FILTER_CONFIG = FilterConfig()
TECHNICAL_CONFIG = TechnicalConfig()
BACKTEST_CONFIG = BacktestConfig()
REPORT_CONFIG = ReportConfig()

ASHARE_POLICY_SECTORS = [
    "高端制造",
    "国产替代",
    "半导体",
    "芯片",
    "AI",
    "数字经济",
    "新能源",
    "新材料",
    "创新药",
    "军工",
    "新基建",
]

US_POLICY_SECTORS = [
    "AI算力",
    "大模型",
    "云计算",
    "SaaS",
    "半导体",
    "硬件科技",
    "互联网",
    "新能源科技",
    "创新药",
]

HK_POLICY_SECTORS = [
    "科技龙头",
    "互联网",
    "创新药",
    "消费龙头",
]

BLACKLIST_PATTERNS = ["ST", "*ST", "退市", "S*ST", "SST"]

# 主要指数配置
INDEX_CONFIG = {
    # 美股指数
    "标普500": {
        "symbol": "SP500",
        "market": "美股",
        "source": "yfinance"
    },
    "纳斯达克综合指数": {
        "symbol": "^IXIC",
        "market": "美股",
        "source": "yfinance"
    },
    # A股指数
    "沪深300": {
        "symbol": "000300",
        "market": "A股",
        "source": "akshare"
    },
    "中证A50": {
        "symbol": "932816",
        "market": "A股",
        "source": "akshare"
    },
    "科创板指数": {
        "symbol": "000688",
        "market": "A股",
        "source": "akshare"
    },
    "创业板指数": {
        "symbol": "399006",
        "market": "A股",
        "source": "akshare"
    }
}

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "quantitative_system.log"
