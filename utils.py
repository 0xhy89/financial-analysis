import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Union, List, Dict, Any
from functools import wraps
import time

import pandas as pd
import numpy as np


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: str = "INFO"
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay * (attempt + 1))
            return None
        return wrapper
    return decorator


def format_market_cap(value: float) -> str:
    if value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f}万亿"
    elif value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}亿"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.2f}万"
    else:
        return f"{value:.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    return f"{value * 100:.{decimals}f}%"


def validate_date(date_str: str) -> bool:
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def get_trading_days(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime]
) -> List[datetime]:
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    days = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def calculate_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change()


def calculate_cumulative_returns(prices: pd.Series) -> pd.Series:
    return (1 + prices.pct_change()).cumprod() - 1


def normalize_code(code: str, market: str) -> str:
    code = code.strip().upper()

    if market == "A股":
        if not code.startswith("SH") and not code.startswith("SZ"):
            if code.startswith("6"):
                code = "SH" + code
            else:
                code = "SZ" + code
    elif market == "港股":
        if not code.startswith("HK"):
            code = "HK" + code
    elif market == "美股":
        if not code.endswith(".US") and not code.startswith("^"):
            code = code + ".US"

    return code


def is_valid_code(code: str, market: str) -> bool:
    code = code.strip().upper()

    if market == "A股":
        return len(code) == 6 and code.isdigit()
    elif market == "港股":
        return len(code) == 5 and code.isdigit()
    elif market == "美股":
        return len(code) >= 1 and (code.replace(".", "").replace("-", "").isalnum())

    return False


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.03,
    periods_per_year: int = 252
) -> float:
    excess_returns = returns - risk_free_rate / periods_per_year
    if excess_returns.std() == 0:
        return 0.0
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()


def calculate_max_drawdown(prices: pd.Series) -> float:
    cumulative = (1 + prices.pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def calculate_win_rate(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    winning_trades = trades[trades["profit"] > 0]
    return len(winning_trades) / len(trades)


def calculate_profit_loss_ratio(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    wins = trades[trades["profit"] > 0]["profit"]
    losses = trades[trades["profit"] < 0]["profit"]

    if len(losses) == 0 or wins.mean() == 0:
        return 0.0

    return abs(wins.mean() / losses.mean())


def save_to_csv(df: pd.DataFrame, filepath: Path, index: bool = False) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=index, encoding="utf-8-sig")


def save_to_json(data: Dict, filepath: Path) -> None:
    import json
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def load_from_csv(filepath: Path) -> pd.DataFrame:
    return pd.read_csv(filepath, encoding="utf-8-sig", index_col=0)


def load_from_json(filepath: Path) -> Dict:
    import json
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    return Path(__file__).parent


def merge_dataframes(
    dataframes: List[pd.DataFrame],
    how: str = "inner"
) -> pd.DataFrame:
    if not dataframes:
        return pd.DataFrame()

    result = dataframes[0]
    for df in dataframes[1:]:
        result = pd.merge(result, df, how=how, left_index=True, right_index=True)

    return result


def fill_missing_values(
    df: pd.DataFrame,
    method: str = "ffill"
) -> pd.DataFrame:
    return df.fillna(method=method)


def remove_outliers(
    series: pd.Series,
    n_std: float = 3.0
) -> pd.Series:
    mean = series.mean()
    std = series.std()
    lower_bound = mean - n_std * std
    upper_bound = mean + n_std * std
    return series.clip(lower=lower_bound, upper=upper_bound)


def safe_divide(a: Union[float, pd.Series], b: Union[float, pd.Series]) -> Union[float, pd.Series]:
    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        return a / b if b != 0 else 0
    return a / b if b != 0 else 0


def get_date_range(period: str) -> tuple[datetime, datetime]:
    end_date = datetime.now()

    period_map = {
        "1y": 1,
        "3y": 3,
        "5y": 5,
        "10y": 10
    }

    years = period_map.get(period, 1)
    start_date = end_date - timedelta(days=years * 365)

    return start_date, end_date


def format_report_date(date: Optional[datetime] = None) -> str:
    if date is None:
        date = datetime.now()
    return date.strftime("%Y年%m月%d日 %H:%M")


def calculate_annual_return(total_return: float, years: float) -> float:
    if years <= 0 or total_return <= -1:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1


def is_market_open(market: str = "A股") -> bool:
    now = datetime.now()
    if now.weekday() >= 5:
        return False

    if market == "A股":
        market_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=0, second=0, microsecond=0)
        return market_time <= now <= market_end

    return True
