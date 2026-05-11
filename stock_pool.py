from datetime import datetime
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import numpy as np

from config import (
    FILTER_CONFIG,
    ASHARE_POLICY_SECTORS,
    US_POLICY_SECTORS,
    HK_POLICY_SECTORS,
    BLACKLIST_PATTERNS,
    LOG_LEVEL,
    LOG_FILE,
    REPORTS_DIR
)
from utils import (
    setup_logger,
    save_to_csv,
    load_from_csv,
    format_market_cap,
    format_percentage,
    normalize_code
)


logger = setup_logger("StockPool", LOG_FILE, LOG_LEVEL)


@dataclass
class StockInfo:
    code: str
    name: str
    market: str
    sector: str
    policy_match: float = 0.0
    pe: float = 0.0
    pb: float = 0.0
    roe: float = 0.0
    market_cap: float = 0.0
    dividend_yield: float = 0.0
    is_st: bool = False
    is_high_position: bool = False
    is_high_pledge: bool = False
    is_high_goodwill: bool = False
    financial_score: float = 0.0
    technical_score: float = 0.0
    overall_score: float = 0.0


@dataclass
class StockPool:
    name: str
    market: str
    stocks: List[StockInfo] = field(default_factory=list)

    def add_stock(self, stock: StockInfo) -> None:
        self.stocks.append(stock)

    def get_stock(self, code: str) -> Optional[StockInfo]:
        for stock in self.stocks:
            if stock.code == code:
                return stock
        return None

    def filter_by_criteria(self, **criteria) -> List[StockInfo]:
        filtered = self.stocks

        for key, value in criteria.items():
            if hasattr(StockInfo, key):
                filtered = [s for s in filtered if getattr(s, key, None) == value]

        return filtered

    def to_dataframe(self) -> pd.DataFrame:
        if not self.stocks:
            return pd.DataFrame()

        data = []
        for stock in self.stocks:
            data.append({
                "code": stock.code,
                "name": stock.name,
                "market": stock.market,
                "sector": stock.sector,
                "policy_match": format_percentage(stock.policy_match),
                "pe": f"{stock.pe:.2f}",
                "pb": f"{stock.pb:.2f}",
                "roe": format_percentage(stock.roe / 100),
                "market_cap": format_market_cap(stock.market_cap),
                "dividend_yield": format_percentage(stock.dividend_yield / 100),
                "is_st": stock.is_st,
                "is_high_position": stock.is_high_position,
                "financial_score": f"{stock.financial_score:.2f}",
                "technical_score": f"{stock.technical_score:.2f}",
                "overall_score": f"{stock.overall_score:.2f}",
            })

        return pd.DataFrame(data)

    def save(self, filepath: Optional[Path] = None) -> None:
        if filepath is None:
            filepath = REPORTS_DIR / f"{self.market}_{self.name}_pool.csv"

        df = self.to_dataframe()
        save_to_csv(df, filepath)
        logger.info(f"Saved stock pool to {filepath}")


class PolicyFilter:
    def __init__(self):
        self.blacklist_patterns = BLACKLIST_PATTERNS

    def is_blacklisted(self, name: str) -> bool:
        name_upper = name.upper()
        for pattern in self.blacklist_patterns:
            if pattern.upper() in name_upper:
                return True
        return False

    def match_policy_sectors(self, name: str, sector: str, market: str) -> float:
        policy_sectors = self._get_policy_sectors(market)

        name_lower = name.lower()
        sector_lower = sector.lower()

        match_count = 0
        for ps in policy_sectors:
            ps_lower = ps.lower()
            if ps_lower in name_lower or ps_lower in sector_lower:
                match_count += 1

        if not policy_sectors:
            return 0.0

        return min(match_count / len(policy_sectors), 1.0)

    def _get_policy_sectors(self, market: str) -> List[str]:
        if market == "A股":
            return ASHARE_POLICY_SECTORS
        elif market == "美股":
            return US_POLICY_SECTORS
        elif market == "港股":
            return HK_POLICY_SECTORS
        return []


class FundamentalFilter:
    def __init__(self, config: type = FILTER_CONFIG):
        self.config = config

    def check_pe(self, pe: float) -> bool:
        if pe <= 0:
            return False
        return self.config.min_pe <= pe <= self.config.max_pe

    def check_pb(self, pb: float) -> bool:
        if pb <= 0:
            return False
        return pb <= self.config.max_pb

    def check_roe(self, roe: float) -> bool:
        return roe >= self.config.min_roe

    def check_market_cap(self, market_cap: float) -> bool:
        return market_cap >= self.config.min_market_cap

    def check_dividend(self, dividend_yield: float) -> bool:
        return dividend_yield > 0

    def check_shares_pledge(self, pledge_ratio: float) -> bool:
        return pledge_ratio <= self.config.max_shares_pledge

    def check_goodwill(self, goodwill_ratio: float) -> bool:
        return goodwill_ratio <= self.config.max_goodwill_ratio

    def check_st_status(self, is_st: bool) -> bool:
        if self.config.exclude_st:
            return not is_st
        return True

    def calculate_financial_score(
        self,
        pe: float,
        pb: float,
        roe: float,
        market_cap: float,
        dividend_yield: float
    ) -> float:
        score = 0.0

        if 0 < pe <= 20:
            score += 30
        elif 20 < pe <= 30:
            score += 20
        elif 30 < pe <= 50:
            score += 10

        if 0 < pb <= 2:
            score += 25
        elif 2 < pb <= 5:
            score += 15
        elif 5 < pb <= 10:
            score += 5

        if roe >= 20:
            score += 25
        elif roe >= 10:
            score += 15
        elif roe >= 5:
            score += 10

        if market_cap >= 100_000_000_000:
            score += 10
        elif market_cap >= 50_000_000_000:
            score += 5

        if dividend_yield >= 3:
            score += 10
        elif dividend_yield >= 1:
            score += 5

        return min(score, 100)


class StockPoolBuilder:
    def __init__(self):
        self.policy_filter = PolicyFilter()
        self.fundamental_filter = FundamentalFilter()
        self._watched_pools: Dict[str, StockPool] = {}

    def build_ashare_pool(
        self,
        stock_list: List[Dict],
        technical_scores: Optional[Dict[str, float]] = None
    ) -> StockPool:
        pool = StockPool(name="政策优选", market="A股")

        technical_scores = technical_scores or {}

        for stock_data in stock_list:
            try:
                stock = self._process_stock_data(stock_data, "A股", technical_scores)

                if stock and self._passes_all_filters(stock):
                    pool.add_stock(stock)

            except Exception as e:
                logger.error(f"Failed to process A股 stock {stock_data.get('code', 'Unknown')}: {e}")
                continue

        self._rank_stocks(pool)
        self._watched_pools["A股"] = pool

        logger.info(f"Built A股 pool with {len(pool.stocks)} stocks")
        return pool

    def build_us_pool(
        self,
        stock_list: List[Dict],
        technical_scores: Optional[Dict[str, float]] = None
    ) -> StockPool:
        pool = StockPool(name="科技主线", market="美股")

        technical_scores = technical_scores or {}

        for stock_data in stock_list:
            try:
                stock = self._process_stock_data(stock_data, "美股", technical_scores)

                if stock and self._passes_all_filters(stock):
                    pool.add_stock(stock)

            except Exception as e:
                logger.error(f"Failed to process 美股 stock {stock_data.get('code', 'Unknown')}: {e}")
                continue

        self._rank_stocks(pool)
        self._watched_pools["美股"] = pool

        logger.info(f"Built 美股 pool with {len(pool.stocks)} stocks")
        return pool

    def build_hk_pool(
        self,
        stock_list: List[Dict],
        technical_scores: Optional[Dict[str, float]] = None
    ) -> StockPool:
        pool = StockPool(name="优质核心资产", market="港股")

        technical_scores = technical_scores or {}

        for stock_data in stock_list:
            try:
                stock = self._process_stock_data(stock_data, "港股", technical_scores)

                if stock and self._passes_all_filters(stock):
                    pool.add_stock(stock)

            except Exception as e:
                logger.error(f"Failed to process 港股 stock {stock_data.get('code', 'Unknown')}: {e}")
                continue

        self._rank_stocks(pool)
        self._watched_pools["港股"] = pool

        logger.info(f"Built 港股 pool with {len(pool.stocks)} stocks")
        return pool

    def _process_stock_data(
        self,
        stock_data: Dict,
        market: str,
        technical_scores: Dict[str, float]
    ) -> Optional[StockInfo]:
        code = stock_data.get("code", "")
        name = stock_data.get("name", "")
        sector = stock_data.get("sector", "")

        if not code:
            return None

        if self.policy_filter.is_blacklisted(name):
            logger.debug(f"Filtered out blacklisted stock: {name}")
            return None

        policy_match = self.policy_filter.match_policy_sectors(name, sector, market)

        financial_data = stock_data.get("financial", {})

        pe = financial_data.get("pe", 0)
        pb = financial_data.get("pb", 0)
        roe = financial_data.get("roe", 0)
        market_cap = financial_data.get("market_cap", 0)
        dividend_yield = financial_data.get("dividend_yield", 0)
        is_st = financial_data.get("is_st", False)

        stock = StockInfo(
            code=code,
            name=name,
            market=market,
            sector=sector,
            policy_match=policy_match,
            pe=pe,
            pb=pb,
            roe=roe,
            market_cap=market_cap,
            dividend_yield=dividend_yield,
            is_st=is_st,
            is_high_position=stock_data.get("is_high_position", False),
            is_high_pledge=stock_data.get("is_high_pledge", False),
            is_high_goodwill=stock_data.get("is_high_goodwill", False)
        )

        stock.financial_score = self.fundamental_filter.calculate_financial_score(
            pe, pb, roe, market_cap, dividend_yield
        )

        stock.technical_score = technical_scores.get(code, 50.0)

        return stock

    def _passes_all_filters(self, stock: StockInfo) -> bool:
        if not self.fundamental_filter.check_st_status(stock.is_st):
            return False

        if stock.is_high_position and FILTER_CONFIG.exclude_high_position:
            return False

        if stock.is_high_pledge:
            return False

        if stock.is_high_goodwill:
            return False

        if stock.policy_match < 0.2:
            return False

        return True

    def _rank_stocks(self, pool: StockPool) -> None:
        for stock in pool.stocks:
            stock.overall_score = (
                stock.policy_match * 30 +
                stock.financial_score * 40 +
                stock.technical_score * 30
            )

        pool.stocks.sort(key=lambda s: s.overall_score, reverse=True)

    def get_pool(self, market: str) -> Optional[StockPool]:
        return self._watched_pools.get(market)

    def get_top_stocks(self, market: str, top_n: int = 10) -> List[StockInfo]:
        pool = self.get_pool(market)
        if pool is None:
            return []
        return pool.stocks[:top_n]

    def get_watched_pools(self) -> Dict[str, StockPool]:
        return self._watched_pools.copy()


class StockPoolManager:
    def __init__(self):
        self.builder = StockPoolBuilder()
        self._pools: Dict[str, StockPool] = {}

    def add_pool(self, pool: StockPool) -> None:
        self._pools[pool.market] = pool

    def get_pool(self, market: str) -> Optional[StockPool]:
        return self._pools.get(market)

    def get_all_stocks(self) -> List[StockInfo]:
        all_stocks = []
        for pool in self._pools.values():
            all_stocks.extend(pool.stocks)
        return all_stocks

    def save_all_pools(self) -> None:
        for market, pool in self._pools.items():
            pool.save()
        logger.info(f"Saved {len(self._pools)} stock pools")

    def generate_pool_report(self) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append("全市场智能量化选股 - 股票池报告")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)

        for market, pool in self._pools.items():
            lines.append(f"\n{'=' * 40}")
            lines.append(f"市场: {market} | 股票池: {pool.name}")
            lines.append(f"股票数量: {len(pool.stocks)}")
            lines.append("=" * 40)

            if pool.stocks:
                lines.append("\nTop 10 推荐标的:")
                lines.append("-" * 80)

                header = f"{'排名':<4} {'代码':<10} {'名称':<15} {'政策匹配':<10} {'财务评分':<10} {'技术评分':<10} {'综合评分':<10} {'所属赛道':<15}"
                lines.append(header)
                lines.append("-" * 80)

                for i, stock in enumerate(pool.stocks[:10], 1):
                    line = (
                        f"{i:<4} "
                        f"{stock.code:<10} "
                        f"{stock.name:<15} "
                        f"{stock.policy_match:.1%}   "
                        f"{stock.financial_score:.1f}   "
                        f"{stock.technical_score:.1f}   "
                        f"{stock.overall_score:.1f}   "
                        f"{stock.sector:<15}"
                    )
                    lines.append(line)

        return "\n".join(lines)


if __name__ == "__main__":
    manager = StockPoolManager()
    builder = StockPoolBuilder()

    mock_stocks = [
        {
            "code": "600519",
            "name": "贵州茅台",
            "sector": "白酒",
            "financial": {
                "pe": 35.0,
                "pb": 12.0,
                "roe": 25.0,
                "market_cap": 250_000_000_000,
                "dividend_yield": 2.5,
                "is_st": False
            }
        },
        {
            "code": "002475",
            "name": "立讯精密",
            "sector": "消费电子",
            "financial": {
                "pe": 25.0,
                "pb": 5.0,
                "roe": 18.0,
                "market_cap": 180_000_000_000,
                "dividend_yield": 1.5,
                "is_st": False
            }
        },
        {
            "code": "000001",
            "name": "平安银行",
            "sector": "银行",
            "financial": {
                "pe": 8.0,
                "pb": 0.8,
                "roe": 12.0,
                "market_cap": 250_000_000_000,
                "dividend_yield": 3.5,
                "is_st": False
            }
        }
    ]

    technical_scores = {
        "600519": 75.0,
        "002475": 82.0,
        "000001": 65.0
    }

    pool = builder.build_ashare_pool(mock_stocks, technical_scores)

    manager.add_pool(pool)

    report = manager.generate_pool_report()
    print(report)
