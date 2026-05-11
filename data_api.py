from datetime import datetime, timedelta
from typing import Optional, List, Dict, Union
from pathlib import Path
import warnings

import pandas as pd
import numpy as np

from config import (
    DATA_DIR,
    DATA_CONFIG,
    LOG_LEVEL,
    LOG_FILE,
    MarketConfig
)
from utils import (
    setup_logger,
    retry,
    save_to_csv,
    load_from_csv,
    ensure_dir,
    normalize_code,
    is_valid_code,
    get_date_range
)


warnings.filterwarnings("ignore")


logger = setup_logger("DataAPI", LOG_FILE, LOG_LEVEL)


class DataCache:
    def __init__(self, cache_dir: Path = DATA_DIR):
        self.cache_dir = cache_dir
        ensure_dir(self.cache_dir)

    def get_cache_path(self, symbol: str, market: str) -> Path:
        return self.cache_dir / f"{market}_{symbol}.csv"

    def is_cache_valid(self, symbol: str, market: str, days: int = 1) -> bool:
        cache_path = self.get_cache_path(symbol, market)
        if not cache_path.exists():
            return False

        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return file_age < timedelta(days=days)

    def save_cache(self, df: pd.DataFrame, symbol: str, market: str) -> None:
        cache_path = self.get_cache_path(symbol, market)
        save_to_csv(df, cache_path)

    def load_cache(self, symbol: str, market: str) -> Optional[pd.DataFrame]:
        cache_path = self.get_cache_path(symbol, market)
        if cache_path.exists():
            try:
                return load_from_csv(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cache for {symbol}: {e}")
        return None


class MarketDataFetcher:
    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache and DATA_CONFIG.cache_enabled
        self.cache = DataCache() if self.use_cache else None
        self._available_modules = self._check_available_modules()

    def _check_available_modules(self) -> Dict[str, bool]:
        modules = {}
        try:
            import akshare
            modules["akshare"] = True
        except ImportError:
            modules["akshare"] = False

        try:
            import yfinance
            modules["yfinance"] = True
        except ImportError:
            modules["yfinance"] = False

        return modules

    @retry(max_attempts=DATA_CONFIG.retry_times, delay=DATA_CONFIG.retry_delay)
    def fetch_ashare_daily(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        if self.use_cache and self.cache.is_cache_valid(symbol, "A股"):
            cached = self.cache.load_cache(symbol, "A股")
            if cached is not None and not cached.empty:
                logger.info(f"Using cached data for A股 {symbol}")
                return cached

        if not self._available_modules.get("akshare", False):
            logger.warning("akshare not available, using mock data for A股")
            return self._generate_mock_data(symbol, "A股", start_date, end_date)

        try:
            import akshare as ak

            symbol_normalized = normalize_code(symbol, "A股")

            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date or (datetime.now() - timedelta(days=365)).strftime("%Y%m%d"),
                end_date=end_date or datetime.now().strftime("%Y%m%d"),
                adjust="qfq"
            )

            if df is not None and not df.empty:
                df = self._normalize_ashare_data(df)
                if self.use_cache:
                    self.cache.save_cache(df, symbol, "A股")
                return df

        except Exception as e:
            logger.error(f"Failed to fetch A股 {symbol}: {e}")

        return self._generate_mock_data(symbol, "A股", start_date, end_date)

    def _normalize_ashare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        column_mapping = {
            "日期": "date",
            "股票代码": "code",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "turnover",
            "振幅": "amplitude",
            "涨跌幅": "pct_change",
            "涨跌额": "change",
            "换手率": "turnover_rate"
        }

        df = df.rename(columns=column_mapping)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

        numeric_columns = ["open", "high", "low", "close", "volume", "turnover", "pct_change"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    @retry(max_attempts=DATA_CONFIG.retry_times, delay=DATA_CONFIG.retry_delay)
    def fetch_hk_daily(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        if self.use_cache and self.cache.is_cache_valid(symbol, "港股"):
            cached = self.cache.load_cache(symbol, "港股")
            if cached is not None and not cached.empty:
                logger.info(f"Using cached data for 港股 {symbol}")
                return cached

        if not self._available_modules.get("akshare", False):
            logger.warning("akshare not available, using mock data for 港股")
            return self._generate_mock_data(symbol, "港股", start_date, end_date)

        try:
            import akshare as ak

            df = ak.stock_hk_daily(symbol=symbol, adjust="qfq")

            if df is not None and not df.empty:
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    if start_date:
                        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if "-" in str(start_date) else datetime.strptime(start_date, "%Y%m%d")
                        df = df[df["date"] >= start_dt]
                    if end_date:
                        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if "-" in str(end_date) else datetime.strptime(end_date, "%Y%m%d")
                        df = df[df["date"] <= end_dt]
                    df = df.set_index("date").sort_index()

                df = self._normalize_hk_data(df)
                if self.use_cache:
                    self.cache.save_cache(df, symbol, "港股")
                return df

        except Exception as e:
            logger.error(f"Failed to fetch 港股 {symbol}: {e}")

        return self._generate_mock_data(symbol, "港股", start_date, end_date)

    def _normalize_hk_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

        numeric_columns = ["open", "high", "low", "close", "volume"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    @retry(max_attempts=DATA_CONFIG.retry_times, delay=DATA_CONFIG.retry_delay)
    def fetch_us_daily(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        if self.use_cache and self.cache.is_cache_valid(symbol, "美股"):
            cached = self.cache.load_cache(symbol, "美股")
            if cached is not None and not cached.empty:
                logger.info(f"Using cached data for 美股 {symbol}")
                return cached

        if not self._available_modules.get("yfinance", False):
            logger.warning("yfinance not available, using mock data for 美股")
            return self._generate_mock_data(symbol, "美股", start_date, end_date)

        try:
            import yfinance as yf

            symbol_normalized = symbol.replace(".US", "") + ".US" if not symbol.endswith(".US") else symbol

            ticker = yf.Ticker(symbol_normalized)
            df = ticker.history(
                start=start_date or (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                end=end_date or datetime.now().strftime("%Y-%m-%d")
            )

            if df is not None and not df.empty:
                df = self._normalize_us_data(df, symbol)
                if self.use_cache:
                    self.cache.save_cache(df, symbol, "美股")
                return df

        except Exception as e:
            logger.error(f"Failed to fetch 美股 {symbol}: {e}")

        return self._generate_mock_data(symbol, "美股", start_date, end_date)

    def _normalize_us_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Dividends": "dividend",
            "Stock Splits": "split"
        })

        if "Close" in df.columns:
            df["pct_change"] = df["Close"].pct_change()

        df.index.name = "date"
        df["code"] = symbol

        return df

    def _generate_mock_data(
        self,
        symbol: str,
        market: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        if start_date:
            start = datetime.strptime(start_date, "%Y%m%d") if len(start_date) == 8 else datetime.strptime(start_date, "%Y-%m-%d")
        else:
            start = datetime.now() - timedelta(days=365)

        if end_date:
            end = datetime.strptime(end_date, "%Y%m%d") if len(end_date) == 8 else datetime.strptime(end_date, "%Y-%m-%d")
        else:
            end = datetime.now()

        dates = pd.date_range(start=start, end=end, freq="B")

        np.random.seed(hash(symbol) % (2**31))

        base_price = 100.0 if market != "美股" else 150.0
        price_data = base_price * (1 + np.random.randn(len(dates)) * 0.02).cumprod()

        df = pd.DataFrame({
            "date": dates,
            "code": symbol,
            "open": price_data * (1 + np.random.randn(len(dates)) * 0.005),
            "high": price_data * (1 + np.abs(np.random.randn(len(dates)) * 0.01)),
            "low": price_data * (1 - np.abs(np.random.randn(len(dates)) * 0.01)),
            "close": price_data,
            "volume": np.random.randint(1_000_000, 10_000_000, len(dates)),
            "pct_change": np.random.randn(len(dates)) * 0.02
        })

        df = df.set_index("date")

        return df

    def fetch_batch(
        self,
        symbols: List[str],
        market: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        results = {}

        for symbol in symbols:
            try:
                if market == "A股":
                    df = self.fetch_ashare_daily(symbol, start_date, end_date)
                elif market == "港股":
                    df = self.fetch_hk_daily(symbol, start_date, end_date)
                elif market == "美股":
                    df = self.fetch_us_daily(symbol, start_date, end_date)
                else:
                    logger.warning(f"Unknown market: {market}")
                    continue

                if df is not None and not df.empty:
                    results[symbol] = df

            except Exception as e:
                logger.error(f"Failed to fetch {market} {symbol}: {e}")
                continue

        return results

    def fetch_index_constituents(self, index_code: str, market: str) -> List[str]:
        if not self._available_modules.get("akshare", False):
            logger.warning("akshare not available, returning empty list")
            return []

        try:
            import akshare as ak

            if market == "A股":
                if index_code.startswith("000") or index_code.startswith("399"):
                    df = ak.index_stock_info(symbol=index_code)
                    if df is not None and "品种代码" in df.columns:
                        return df["品种代码"].tolist()

        except Exception as e:
            logger.error(f"Failed to fetch index constituents for {index_code}: {e}")

        return []

    def fetch_market_overview(self, market: str) -> pd.DataFrame:
        if not self._available_modules.get("akshare", False):
            logger.warning("akshare not available")
            return pd.DataFrame()

        try:
            import akshare as ak

            if market == "A股":
                df = ak.stock_zh_a_spot_em()
                if df is not None:
                    return df

        except Exception as e:
            logger.error(f"Failed to fetch market overview for {market}: {e}")

        return pd.DataFrame()


class FundamentalDataFetcher:
    def __init__(self):
        self._available = self._check_akshare()

    def _check_akshare(self) -> bool:
        try:
            import akshare
            return True
        except ImportError:
            return False

    def fetch_financial_data(self, symbol: str, market: str) -> Dict:
        if not self._available:
            return self._get_mock_financial_data(symbol)

        try:
            import akshare as ak

            if market == "A股":
                return self._fetch_ashare_financial(symbol)
            elif market == "港股":
                return self._fetch_hk_financial(symbol)
            elif market == "美股":
                return self._fetch_us_financial(symbol)

        except Exception as e:
            logger.error(f"Failed to fetch financial data for {symbol}: {e}")

        return self._get_mock_financial_data(symbol)

    def _fetch_ashare_financial(self, symbol: str) -> Dict:
        try:
            import akshare as ak

            financial_data = {}

            try:
                df = ak.stock_financial_analysis_indicator(symbol=symbol)
                if df is not None and not df.empty:
                    latest = df.iloc[-1]
                    financial_data["roe"] = latest.get("净资产收益率(%)", 0)
                    financial_data["pe"] = latest.get("市盈率", 0)
                    financial_data["pb"] = latest.get("市净率", 0)
            except:
                pass

            return financial_data

        except Exception:
            return self._get_mock_financial_data(symbol)

    def _fetch_hk_financial(self, symbol: str) -> Dict:
        return self._get_mock_financial_data(symbol)

    def _fetch_us_financial(self, symbol: str) -> Dict:
        if not self._available:
            return self._get_mock_financial_data(symbol)

        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol + ".US" if not symbol.endswith(".US") else symbol)
            info = ticker.info

            return {
                "pe": info.get("trailingPE", 0),
                "pb": info.get("priceToBook", 0),
                "roe": info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else 0,
                "market_cap": info.get("marketCap", 0),
            }

        except Exception:
            return self._get_mock_financial_data(symbol)

    def _get_mock_financial_data(self, symbol: str) -> Dict:
        np.random.seed(hash(symbol) % (2**31))

        return {
            "pe": np.random.uniform(5, 30),
            "pb": np.random.uniform(0.5, 5),
            "roe": np.random.uniform(5, 25),
            "market_cap": np.random.uniform(10_000_000_000, 500_000_000_000),
            "dividend_yield": np.random.uniform(0, 5),
            "debt_ratio": np.random.uniform(0.2, 0.7),
        }


class DataAPI:
    def __init__(self, use_cache: bool = True):
        self.market_fetcher = MarketDataFetcher(use_cache)
        self.fundamental_fetcher = FundamentalDataFetcher()
        self.use_cache = use_cache

    def get_daily_data(
        self,
        symbol: str,
        market: str,
        period: str = "1y"
    ) -> pd.DataFrame:
        start_date, end_date = get_date_range(period)

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        if market == "A股":
            return self.market_fetcher.fetch_ashare_daily(symbol, start_str, end_str)
        elif market == "港股":
            return self.market_fetcher.fetch_hk_daily(symbol, start_str, end_str)
        elif market == "美股":
            return self.market_fetcher.fetch_us_daily(symbol, start_str, end_str)

        return pd.DataFrame()

    def get_financial_data(self, symbol: str, market: str) -> Dict:
        return self.fundamental_fetcher.fetch_financial_data(symbol, market)

    def get_batch_data(
        self,
        symbols: List[str],
        market: str,
        period: str = "1y"
    ) -> Dict[str, pd.DataFrame]:
        start_date, end_date = get_date_range(period)

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        return self.market_fetcher.fetch_batch(symbols, market, start_str, end_str)

    def get_market_overview(self, market: str) -> pd.DataFrame:
        return self.market_fetcher.fetch_market_overview(market)

    def get_index_constituents(self, index_code: str, market: str) -> List[str]:
        return self.market_fetcher.fetch_index_constituents(index_code, market)


if __name__ == "__main__":
    api = DataAPI(use_cache=False)

    test_symbols = ["000001", "600519", "00700"]

    for i, symbol in enumerate(test_symbols):
        market = ["A股", "A股", "港股"][i]
        print(f"\nFetching {market} {symbol}...")

        df = api.get_daily_data(symbol, market, "3mo")
        if not df.empty:
            print(f"  Data shape: {df.shape}")
            print(f"  Latest close: {df['close'].iloc[-1]:.2f}")
