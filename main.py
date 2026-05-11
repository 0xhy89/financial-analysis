import sys
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

from config import (
    BASE_DIR,
    DATA_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    LOG_LEVEL,
    LOG_FILE,
    MARKETS,
    ASHARE_POLICY_SECTORS,
    US_POLICY_SECTORS,
    HK_POLICY_SECTORS,
    BACKTEST_CONFIG
)
from utils import (
    setup_logger,
    save_to_csv,
    get_date_range,
    format_report_date
)
from data_api import DataAPI
from stock_pool import (
    StockPool,
    StockPoolBuilder,
    StockPoolManager,
    StockInfo
)
from strategy import StrategyType, StrategyEngine
from daily_recommend import (
    DailyRecommendSystem,
    DailyRecommendation,
    ReportGenerator
)
from backtest import (
    BacktestEngine,
    BacktestResult,
    BacktestReportGenerator,
    PortfolioBacktester
)


logger = setup_logger("Main", LOG_FILE, LOG_LEVEL)


class QuantitativeSystem:
    def __init__(self, use_cache: bool = True):
        self.data_api = DataAPI(use_cache=use_cache)
        self.pool_builder = StockPoolBuilder()
        self.pool_manager = StockPoolManager()
        self.recommend_system = DailyRecommendSystem()
        self.backtest_engine = BacktestEngine()
        self.portfolio_backtester = PortfolioBacktester()
        self.report_generator = ReportGenerator()
        self.backtest_report_generator = BacktestReportGenerator()

    def build_stock_pools(
        self,
        market: str,
        symbols: Optional[List[str]] = None
    ) -> StockPool:
        logger.info(f"Building {market} stock pool...")

        if symbols is None:
            symbols = self._get_default_symbols(market)

        if not symbols:
            logger.warning(f"No symbols provided for {market}")
            return StockPool(name="Empty", market=market)

        stock_data_list = []

        for symbol in symbols:
            try:
                financial_data = self.data_api.get_financial_data(symbol, market)

                stock_info = {
                    "code": symbol,
                    "name": self._get_stock_name(symbol, market),
                    "sector": self._get_stock_sector(symbol, market),
                    "financial": financial_data
                }

                stock_data_list.append(stock_info)

            except Exception as e:
                logger.error(f"Failed to get financial data for {symbol}: {e}")
                continue

        technical_scores = self._calculate_technical_scores(symbols, market)

        if market == "A股":
            pool = self.pool_builder.build_ashare_pool(stock_data_list, technical_scores)
        elif market == "港股":
            pool = self.pool_builder.build_hk_pool(stock_data_list, technical_scores)
        elif market == "美股":
            pool = self.pool_builder.build_us_pool(stock_data_list, technical_scores)
        else:
            pool = StockPool(name="Unknown", market=market)

        self.pool_manager.add_pool(pool)

        logger.info(f"Built {market} pool with {len(pool.stocks)} stocks")

        return pool

    def _get_default_symbols(self, market: str) -> List[str]:
        defaults = {
            "A股": ["600519", "000001", "002475", "600036", "601318", "000858", "600276", "300750", "002594", "600900"],
            "港股": ["00700", "03690", "09988", "01093", "02318", "00788", "06098", "06618"],
            "美股": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "JNJ", "V"]
        }
        return defaults.get(market, [])

    def _get_stock_name(self, symbol: str, market: str) -> str:
        name_map = {
            "A股": {
                "600519": "贵州茅台", "000001": "平安银行", "002475": "立讯精密",
                "600036": "招商银行", "601318": "中国平安", "000858": "五粮液",
                "600276": "恒瑞医药", "300750": "宁德时代", "002594": "比亚迪",
                "600900": "长江电力"
            },
            "港股": {
                "00700": "腾讯控股", "03690": "美团", "09988": "阿里巴巴",
                "01093": "中国生物制药", "02318": "平安好医生", "00788": "铁塔公司"
            },
            "美股": {
                "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet",
                "AMZN": "Amazon", "NVDA": "NVIDIA", "META": "Meta Platforms",
                "TSLA": "Tesla", "JPM": "JPMorgan", "JNJ": "Johnson & Johnson",
                "V": "Visa"
            }
        }
        return name_map.get(market, {}).get(symbol, symbol)

    def _get_stock_sector(self, symbol: str, market: str) -> str:
        sector_map = {
            "A股": {
                "600519": "白酒", "000001": "银行", "002475": "消费电子",
                "600036": "银行", "601318": "保险", "000858": "白酒",
                "600276": "创新药", "300750": "新能源", "002594": "新能源汽车",
                "600900": "电力"
            },
            "港股": {
                "00700": "互联网", "03690": "本地生活", "09988": "电商",
                "01093": "创新药", "02318": "医疗", "00788": "通信"
            },
            "美股": {
                "AAPL": "科技", "MSFT": "科技", "GOOGL": "科技",
                "AMZN": "电商", "NVDA": "AI/芯片", "META": "社交",
                "TSLA": "新能源", "JPM": "金融", "JNJ": "医疗", "V": "支付"
            }
        }
        return sector_map.get(market, {}).get(symbol, "综合")

    def _calculate_technical_scores(
        self,
        symbols: List[str],
        market: str
    ) -> Dict[str, float]:
        strategy_engine = StrategyEngine()
        scores = {}

        for symbol in symbols:
            try:
                df = self.data_api.get_daily_data(symbol, market, "3mo")
                if not df.empty:
                    score = strategy_engine.score_stock(df, symbol, StrategyType.COMBINED)
                    scores[symbol] = score
            except Exception as e:
                logger.error(f"Failed to calculate technical score for {symbol}: {e}")
                scores[symbol] = 50.0

        return scores

    def generate_daily_recommendations(
        self,
        market_configs: Optional[Dict[str, int]] = None
    ) -> Dict[str, DailyRecommendation]:
        logger.info("Generating daily recommendations...")

        pools = {}
        for market in ["A股", "港股", "美股"]:
            pool = self.pool_manager.get_pool(market)
            if pool:
                pools[market] = pool

        if not pools:
            logger.warning("No stock pools available, building default pools...")
            for market in ["A股", "港股", "美股"]:
                pool = self.build_stock_pools(market)
                pools[market] = pool

        if market_configs is None:
            market_configs = {market: MARKETS[market].recommend_count for market in MARKETS}

        recommendations = self.recommend_system.run(pools, market_configs)

        logger.info("Daily recommendations generated successfully")

        return recommendations

    def run_backtest(
        self,
        symbol: str,
        market: str,
        period: str = "2y",
        strategy_type: StrategyType = StrategyType.COMBINED
    ) -> BacktestResult:
        logger.info(f"Running backtest for {symbol} ({market})...")

        df = self.data_api.get_daily_data(symbol, market, period)

        if df.empty:
            logger.error(f"No data available for backtesting {symbol}")
            return self.backtest_engine._create_empty_result(symbol, strategy_type)

        result = self.backtest_engine.backtest(df, symbol, strategy_type)

        logger.info(f"Backtest completed: Total Return={result.total_return:.2%}")

        return result

    def run_portfolio_backtest(
        self,
        symbols: List[str],
        market: str,
        period: str = "2y",
        strategy_type: StrategyType = StrategyType.COMBINED
    ) -> List[BacktestResult]:
        logger.info(f"Running portfolio backtest for {len(symbols)} {market} stocks...")

        stock_data = {}
        for symbol in symbols:
            df = self.data_api.get_daily_data(symbol, market, period)
            if not df.empty:
                stock_data[symbol] = df

        if not stock_data:
            logger.error("No stock data available for backtest")
            return []

        results = self.portfolio_backtester.backtest_portfolio(
            stock_data, strategy_type
        )

        summary = self.portfolio_backtester.generate_portfolio_summary(results)
        logger.info(f"Portfolio backtest summary: {summary}")

        return results

    def run_full_analysis(
        self,
        markets: List[str] = None,
        generate_recommendations: bool = True,
        run_backtest: bool = False,
        backtest_symbols: Optional[Dict[str, List[str]]] = None
    ) -> Dict:
        if markets is None:
            markets = ["A股", "港股", "美股"]

        if backtest_symbols is None:
            backtest_symbols = {
                "A股": ["600519", "000001", "002475"],
                "港股": ["00700", "03690", "09988"],
                "美股": ["AAPL", "MSFT", "GOOGL"]
            }

        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stock_pools": {},
            "recommendations": {},
            "backtest_results": {}
        }

        logger.info("=" * 80)
        logger.info("Starting Full Quantitative Analysis System")
        logger.info("=" * 80)

        for market in markets:
            logger.info(f"\nProcessing {market}...")

            pool = self.build_stock_pools(market)
            results["stock_pools"][market] = pool.to_dataframe()

            if run_backtest and market in backtest_symbols:
                symbols = backtest_symbols[market]
                backtest_results = self.run_portfolio_backtest(
                    symbols, market, period="2y"
                )
                results["backtest_results"][market] = backtest_results

        if generate_recommendations:
            market_config = {m: MARKETS[m].recommend_count for m in markets}
            recommendations = self.generate_daily_recommendations(market_config)
            results["recommendations"].update(recommendations)

        self._print_summary(results)

        self._save_results(results)

        logger.info("\n" + "=" * 80)
        logger.info("Full Analysis Completed Successfully")
        logger.info("=" * 80)

        return results

    def _print_summary(self, results: Dict) -> None:
        print("\n" + "=" * 80)
        print("分析结果摘要")
        print("=" * 80)

        if "stock_pools" in results:
            print("\n【股票池】")
            for market, df in results["stock_pools"].items():
                print(f"  {market}: {len(df)} 只股票")

        if "recommendations" in results:
            print("\n【每日推荐】")
            for market, rec in results["recommendations"].items():
                if isinstance(rec, dict):
                    for m, r in rec.items():
                        if hasattr(r, 'recommendations'):
                            print(f"  {m}: {len(r.recommendations)} 只推荐标的")
                elif hasattr(rec, 'recommendations'):
                    print(f"  {market}: {len(rec.recommendations)} 只推荐标的")

        if "backtest_results" in results:
            print("\n【回测结果】")
            for market, results_list in results["backtest_results"].items():
                if results_list:
                    avg_return = sum(r.total_return for r in results_list) / len(results_list)
                    avg_sharpe = sum(r.sharpe_ratio for r in results_list) / len(results_list)
                    print(f"  {market}: 平均收益={avg_return:.2%}, 平均夏普比率={avg_sharpe:.2f}")

    def _save_results(self, results: Dict) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        if "stock_pools" in results:
            for market, df in results["stock_pools"].items():
                filepath = REPORTS_DIR / f"stock_pool_{market}_{timestamp}.csv"
                save_to_csv(df, filepath)

        if "recommendations" in results:
            self.report_generator.save_report(results["recommendations"], "txt")
            self.report_generator.save_report(results["recommendations"], "csv")

        if "backtest_results" in results:
            for market, results_list in results["backtest_results"].items():
                if results_list:
                    self.backtest_report_generator.save_report(results_list, market, "txt")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="全市场智能量化选股 + 每日自动推荐 + 策略回测系统"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "pools", "recommend", "backtest"],
        help="运行模式"
    )

    parser.add_argument(
        "--market",
        type=str,
        nargs="+",
        default=["A股", "港股", "美股"],
        choices=["A股", "港股", "美股"],
        help="选择市场"
    )

    parser.add_argument(
        "--period",
        type=str,
        default="2y",
        choices=["1y", "3y", "5y", "10y"],
        help="回测周期"
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="禁用数据缓存"
    )

    parser.add_argument(
        "--backtest-symbols",
        type=str,
        nargs="+",
        help="回测标的代码"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    system = QuantitativeSystem(use_cache=not args.no_cache)

    if args.mode == "full":
        system.run_full_analysis(
            markets=args.market,
            generate_recommendations=True,
            run_backtest=True
        )

    elif args.mode == "pools":
        for market in args.market:
            pool = system.build_stock_pools(market)
            print(f"\n{market} 股票池 ({len(pool.stocks)} 只):")
            if pool.stocks:
                for stock in pool.stocks[:10]:
                    print(f"  {stock.code} {stock.name} - {stock.sector} (评分: {stock.overall_score:.1f})")

    elif args.mode == "recommend":
        recommendations = system.generate_daily_recommendations()
        report = system.report_generator.generate_text_report(recommendations)
        print(report)

    elif args.mode == "backtest":
        if args.backtest_symbols:
            for symbol in args.backtest_symbols:
                market = "A股"
                if "." in symbol:
                    if symbol.endswith(".HK"):
                        market = "港股"
                    elif symbol.endswith(".US"):
                        market = "美股"
                    symbol = symbol.replace(".HK", "").replace(".US", "")

                result = system.run_backtest(symbol, market, args.period)
                print(f"\n{symbol} 回测结果:")
                print(f"  总收益: {result.total_return:.2%}")
                print(f"  年化收益: {result.annual_return:.2%}")
                print(f"  最大回撤: {result.max_drawdown:.2%}")
                print(f"  夏普比率: {result.sharpe_ratio:.2f}")
                print(f"  交易次数: {result.total_trades}")
        else:
            print("请指定回测标的 --backtest-symbols")

    print("\n系统运行完成!")


if __name__ == "__main__":
    main()
