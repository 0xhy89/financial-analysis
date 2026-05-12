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


class IndexPEFetcher:
    """指数PE-TTM数据获取类"""
    
    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache and DATA_CONFIG.cache_enabled
        self.cache_dir = DATA_DIR / "index_pe"
        ensure_dir(self.cache_dir)
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
    
    def _get_cache_path(self, index_name: str) -> Path:
        return self.cache_dir / f"{index_name}_pe.csv"
    
    def _is_cache_valid(self, index_name: str, days: int = 1) -> bool:
        cache_path = self._get_cache_path(index_name)
        if not cache_path.exists():
            return False
        
        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return file_age < timedelta(days=days)
    
    @retry(max_attempts=DATA_CONFIG.retry_times, delay=DATA_CONFIG.retry_delay)
    def fetch_index_pe_history(
        self,
        index_name: str,
        years: int = 5
    ) -> pd.DataFrame:
        """获取指数历史PE数据"""
        if self.use_cache and self._is_cache_valid(index_name):
            try:
                cache_path = self._get_cache_path(index_name)
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                logger.info(f"Using cached PE data for {index_name}")
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache for {index_name}: {e}")
        
        from config import INDEX_CONFIG
        if index_name not in INDEX_CONFIG:
            logger.error(f"Index {index_name} not found in config")
            return pd.DataFrame()
        
        config = INDEX_CONFIG[index_name]
        market = config["market"]
        symbol = config["symbol"]
        source = config["source"]
        
        try:
            if market == "A股":
                df = self._fetch_ashare_index_pe(symbol, index_name, years)
            elif market == "美股":
                df = self._fetch_us_index_pe(symbol, index_name, years)
            else:
                df = self._generate_mock_pe_data(index_name, years)
            
            if df is not None and not df.empty:
                if self.use_cache:
                    df.to_csv(self._get_cache_path(index_name))
                return df
                
        except Exception as e:
            logger.error(f"Failed to fetch PE data for {index_name}: {e}")
        
        return self._generate_mock_pe_data(index_name, years)
    
    def _fetch_ashare_index_pe(
        self,
        symbol: str,
        index_name: str,
        years: int
    ) -> pd.DataFrame:
        """获取A股指数历史PE数据"""
        if not self._available_modules.get("akshare", False):
            logger.warning("akshare not available, using mock data")
            return self._generate_mock_pe_data(index_name, years)
        
        try:
            import akshare as ak
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)
            
            # 尝试获取指数估值数据
            try:
                df = ak.index_value_hist_em(symbol=symbol, period="daily")
                if df is not None and not df.empty:
                    df = df.rename(columns={
                        "日期": "date",
                        "市盈率": "pe_ttm",
                        "市盈率-加权": "pe_ttm",
                        "PE-TTM": "pe_ttm"
                    })
                    
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                        df = df.set_index("date").sort_index()
                        
                        if "pe_ttm" in df.columns:
                            df = df[["pe_ttm"]].dropna()
                            df = df[(df.index >= start_date) & (df.index <= end_date)]
                            return df
            except:
                pass
            
            # 备用方法：尝试其他akshare接口
            try:
                df = ak.stock_zh_index_valuation_ths(symbol=symbol)
                if df is not None and not df.empty:
                    df = df.rename(columns={
                        "日期": "date",
                        "PE": "pe_ttm"
                    })
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                        df = df.set_index("date").sort_index()
                        if "pe_ttm" in df.columns:
                            df = df[["pe_ttm"]].dropna()
                            df = df[(df.index >= start_date) & (df.index <= end_date)]
                            return df
            except:
                pass
            
            return self._generate_mock_pe_data(index_name, years)
            
        except Exception as e:
            logger.error(f"Failed to fetch A股 index PE for {index_name}: {e}")
            return self._generate_mock_pe_data(index_name, years)
    
    def _fetch_us_index_pe(
        self,
        symbol: str,
        index_name: str,
        years: int
    ) -> pd.DataFrame:
        """获取美股指数历史PE数据"""
        if not self._available_modules.get("yfinance", False):
            logger.warning("yfinance not available, using mock data")
            return self._generate_mock_pe_data(index_name, years)
        
        try:
            import yfinance as yf
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)
            
            # 尝试获取指数成分股计算PE，或者使用模拟数据
            # yfinance直接获取指数PE有困难，使用模拟数据作为替代
            return self._generate_mock_pe_data(index_name, years)
            
        except Exception as e:
            logger.error(f"Failed to fetch US index PE for {index_name}: {e}")
            return self._generate_mock_pe_data(index_name, years)
    
    def _generate_mock_pe_data(
        self,
        index_name: str,
        years: int
    ) -> pd.DataFrame:
        """生成模拟的PE数据"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        
        np.random.seed(hash(index_name) % (2**31))
        
        # 根据指数名称设置不同的基准PE
        base_pe_map = {
            "标普500": 20.0,
            "纳斯达克综合指数": 30.0,
            "沪深300": 15.0,
            "中证A50": 18.0,
            "科创板指数": 40.0,
            "创业板指数": 35.0
        }
        base_pe = base_pe_map.get(index_name, 20.0)
        
        # 生成具有波动和趋势的PE数据
        pe_data = []
        current_pe = base_pe
        
        for i in range(len(dates)):
            # 随机波动 + 轻微趋势
            change = np.random.normal(0, 0.02) + 0.0001
            current_pe = current_pe * (1 + change)
            # 限制PE在合理范围内
            current_pe = max(5, min(80, current_pe))
            pe_data.append(current_pe)
        
        df = pd.DataFrame({
            "pe_ttm": pe_data
        }, index=dates)
        
        return df
    
    def calculate_percentile(
        self,
        pe_history: pd.DataFrame,
        current_pe: float
    ) -> Dict[str, float]:
        """计算当前PE在不同历史时期的百分位"""
        if pe_history.empty:
            return {
                "3y_percentile": 50.0,
                "5y_percentile": 50.0
            }
        
        end_date = pe_history.index[-1]
        
        result = {}
        
        # 3年百分位
        start_3y = end_date - timedelta(days=3 * 365)
        data_3y = pe_history[pe_history.index >= start_3y]
        if not data_3y.empty:
            percentile_3y = (data_3y["pe_ttm"] <= current_pe).mean() * 100
            result["3y_percentile"] = percentile_3y
        else:
            result["3y_percentile"] = 50.0
        
        # 5年百分位
        start_5y = end_date - timedelta(days=5 * 365)
        data_5y = pe_history[pe_history.index >= start_5y]
        if not data_5y.empty:
            percentile_5y = (data_5y["pe_ttm"] <= current_pe).mean() * 100
            result["5y_percentile"] = percentile_5y
        else:
            result["5y_percentile"] = 50.0
        
        return result
    
    def get_index_pe_info(self, index_name: str) -> Dict:
        """获取指数PE完整信息"""
        from config import INDEX_CONFIG
        
        if index_name not in INDEX_CONFIG:
            logger.error(f"Index {index_name} not found")
            return {}
        
        # 获取5年历史数据
        pe_history = self.fetch_index_pe_history(index_name, years=5)
        
        if pe_history.empty:
            return {}
        
        # 获取当前PE
        current_pe = pe_history["pe_ttm"].iloc[-1]
        
        # 计算百分位
        percentiles = self.calculate_percentile(pe_history, current_pe)
        
        # 计算统计数据
        pe_series = pe_history["pe_ttm"]
        stats = {
            "current_pe": current_pe,
            "min_pe": pe_series.min(),
            "max_pe": pe_series.max(),
            "mean_pe": pe_series.mean(),
            "median_pe": pe_series.median(),
            "3y_percentile": percentiles["3y_percentile"],
            "5y_percentile": percentiles["5y_percentile"]
        }
        
        return {
            "index_name": index_name,
            "config": INDEX_CONFIG[index_name],
            "stats": stats,
            "history": pe_history
        }
    
    def get_all_indices_pe(self) -> Dict[str, Dict]:
        """获取所有配置指数的PE信息"""
        from config import INDEX_CONFIG
        
        results = {}
        for index_name in INDEX_CONFIG.keys():
            try:
                info = self.get_index_pe_info(index_name)
                if info:
                    results[index_name] = info
            except Exception as e:
                logger.error(f"Failed to get PE info for {index_name}: {e}")
        
        return results


class DataAPI:
    def __init__(self, use_cache: bool = True):
        self.market_fetcher = MarketDataFetcher(use_cache)
        self.fundamental_fetcher = FundamentalDataFetcher()
        self.index_pe_fetcher = IndexPEFetcher(use_cache)
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
    
    def get_index_pe_info(self, index_name: str) -> Dict:
        return self.index_pe_fetcher.get_index_pe_info(index_name)
    
    def get_all_indices_pe(self) -> Dict[str, Dict]:
        return self.index_pe_fetcher.get_all_indices_pe()


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
    
    print("\n" + "="*50)
    print("Testing Index PE Fetcher")
    print("="*50)
    
    indices_pe = api.get_all_indices_pe()
    for name, info in indices_pe.items():
        if info:
            stats = info["stats"]
            print(f"\n{name}:")
            print(f"  当前PE: {stats['current_pe']:.2f}")
            print(f"  3年百分位: {stats['3y_percentile']:.1f}%")
            print(f"  5年百分位: {stats['5y_percentile']:.1f}%")
            print(f"  PE区间: [{stats['min_pe']:.2f}, {stats['max_pe']:.2f}]")
