from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum

import pandas as pd
import numpy as np

from config import (
    BACKTEST_CONFIG,
    REPORTS_DIR,
    LOG_LEVEL,
    LOG_FILE
)
from utils import (
    setup_logger,
    save_to_csv,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_win_rate,
    calculate_profit_loss_ratio,
    calculate_annual_return,
    format_percentage
)
from strategy import TradeSignal, StrategyType, StrategyEngine


logger = setup_logger("Backtest", LOG_FILE, LOG_LEVEL)


class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class Position:
    symbol: str
    entry_date: datetime
    entry_price: float
    quantity: float
    side: PositionSide = PositionSide.LONG

    @property
    def market_value(self) -> float:
        return self.entry_price * self.quantity

    def current_value(self, current_price: float) -> float:
        return current_price * self.quantity


@dataclass
class Trade:
    entry_date: datetime
    exit_date: datetime
    symbol: str
    side: PositionSide
    entry_price: float
    exit_price: float
    quantity: float
    commission: float
    slippage: float
    profit: float
    profit_pct: float

    def to_dict(self) -> Dict:
        return {
            "entry_date": self.entry_date.strftime("%Y-%m-%d"),
            "exit_date": self.exit_date.strftime("%Y-%m-%d"),
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "commission": self.commission,
            "slippage": self.slippage,
            "profit": self.profit,
            "profit_pct": self.profit_pct
        }


@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    initial_cash: float = field(default_factory=lambda: BACKTEST_CONFIG.initial_cash)

    @property
    def total_value(self) -> float:
        position_value = sum(p.current_value if hasattr(p, 'current_value') else 0
                           for p in self.positions.values())
        return self.cash + sum(
            p.entry_price * p.quantity for p in self.positions.values()
        )

    def total_market_value(self) -> float:
        return sum(p.entry_price * p.quantity for p in self.positions.values())


@dataclass
class BacktestResult:
    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_cash: float
    final_value: float
    total_return: float
    annual_return: float
    max_drawdown: float
    win_rate: float
    profit_loss_ratio: float
    sharpe_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_holding_days: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self) -> Dict:
        return {
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "start_date": self.start_date.strftime("%Y-%m-%d"),
            "end_date": self.end_date.strftime("%Y-%m-%d"),
            "initial_cash": self.initial_cash,
            "final_value": self.final_value,
            "total_return": format_percentage(self.total_return),
            "annual_return": format_percentage(self.annual_return),
            "max_drawdown": format_percentage(self.max_drawdown),
            "win_rate": format_percentage(self.win_rate),
            "profit_loss_ratio": f"{self.profit_loss_ratio:.2f}",
            "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "avg_holding_days": f"{self.avg_holding_days:.1f}"
        }


class RiskManager:
    def __init__(self, config: BACKTEST_CONFIG = BACKTEST_CONFIG):
        self.config = config

    def should_take_position(
        self,
        portfolio: Portfolio,
        symbol: str,
        price: float
    ) -> Tuple[bool, float]:
        max_position_value = self.config.initial_cash * self.config.max_position

        current_position_value = 0
        if symbol in portfolio.positions:
            current_position_value = portfolio.positions[symbol].market_value

        if current_position_value >= max_position_value:
            return False, 0

        available_cash = portfolio.cash * self.config.max_total_position

        position_value = min(max_position_value - current_position_value, available_cash)

        if position_value <= 0:
            return False, 0

        shares = int(position_value / price)

        return shares > 0, shares

    def should_close_position(
        self,
        position: Position,
        current_price: float,
        stop_loss: float,
        take_profit: float
    ) -> Tuple[bool, str]:
        if position.side == PositionSide.LONG:
            profit_pct = (current_price - position.entry_price) / position.entry_price
        else:
            profit_pct = (position.entry_price - current_price) / position.entry_price

        if profit_pct <= -stop_loss:
            return True, "STOP_LOSS"

        if profit_pct >= take_profit:
            return True, "TAKE_PROFIT"

        return False, ""

    def check_max_drawdown(self, peak_value: float, current_value: float) -> bool:
        if peak_value <= 0:
            return False

        drawdown = (peak_value - current_value) / peak_value
        return drawdown >= self.config.max_drawdown


class BacktestEngine:
    def __init__(self, config: BACKTEST_CONFIG = BACKTEST_CONFIG):
        self.config = config
        self.risk_manager = RiskManager(config)
        self.strategy_engine = StrategyEngine()

    def backtest(
        self,
        df: pd.DataFrame,
        symbol: str,
        strategy_type: StrategyType = StrategyType.COMBINED,
        initial_cash: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> BacktestResult:
        if df.empty or len(df) < 30:
            logger.warning(f"Insufficient data for backtesting {symbol}")
            return self._create_empty_result(symbol, strategy_type)

        if initial_cash is None:
            initial_cash = self.config.initial_cash
        if stop_loss is None:
            stop_loss = self.config.stop_loss
        if take_profit is None:
            take_profit = self.config.take_profit

        result = BacktestResult(
            strategy_name=strategy_type.value,
            symbol=symbol,
            start_date=pd.to_datetime(df.index[0]).to_pydatetime(),
            end_date=pd.to_datetime(df.index[-1]).to_pydatetime(),
            initial_cash=initial_cash,
            final_value=initial_cash,
            total_return=0.0,
            annual_return=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_loss_ratio=0.0,
            sharpe_ratio=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            avg_holding_days=0.0
        )

        portfolio = Portfolio(cash=initial_cash, initial_cash=initial_cash)

        strategy_result = self.strategy_engine.selector.analyze_with_strategy(
            df, symbol, strategy_type
        )

        signals = strategy_result.signals

        equity_dates = []
        equity_values = []

        peak_value = initial_cash
        current_position: Optional[Position] = None
        trades_list: List[Trade] = []
        holding_days = []

        for i, (date, row) in enumerate(df.iterrows()):
            current_price = row["close"]
            date = pd.to_datetime(date).to_pydatetime()

            equity_dates.append(date)
            equity_values.append(portfolio.total_value)

            if current_position is None:
                buy_signals = [s for s in signals if s.signal_type == "BUY" and s.date == date]

                if buy_signals:
                    should_take, quantity = self.risk_manager.should_take_position(
                        portfolio, symbol, current_price
                    )

                    if should_take:
                        commission = current_price * quantity * self.config.commission
                        slippage_cost = current_price * quantity * self.config.slippage
                        total_cost = commission + slippage_cost

                        current_position = Position(
                            symbol=symbol,
                            entry_date=date,
                            entry_price=current_price,
                            quantity=quantity,
                            side=PositionSide.LONG
                        )

                        portfolio.cash -= (current_price * quantity + total_cost)

            else:
                should_close, close_reason = self.risk_manager.should_close_position(
                    current_position,
                    current_price,
                    stop_loss,
                    take_profit
                )

                if not should_close:
                    sell_signals = [s for s in signals if s.signal_type == "SELL" and s.date == date]
                    if sell_signals:
                        should_close = True
                        close_reason = "SIGNAL_SELL"

                if should_close:
                    commission = current_price * current_position.quantity * self.config.commission
                    slippage_cost = current_price * current_position.quantity * self.config.slippage

                    exit_price = current_price * (1 - self.config.slippage) if close_reason == "STOP_LOSS" else current_price

                    profit = (exit_price - current_position.entry_price) * current_position.quantity - commission - slippage_cost
                    profit_pct = profit / (current_position.entry_price * current_position.quantity)

                    holding_days.append((date - current_position.entry_date).days)

                    trade = Trade(
                        entry_date=current_position.entry_date,
                        exit_date=date,
                        symbol=symbol,
                        side=current_position.side,
                        entry_price=current_position.entry_price,
                        exit_price=exit_price,
                        quantity=current_position.quantity,
                        commission=commission,
                        slippage=slippage_cost,
                        profit=profit,
                        profit_pct=profit_pct
                    )

                    trades_list.append(trade)

                    portfolio.cash += current_position.quantity * exit_price - commission

                    current_position = None

            current_total = portfolio.total_value
            if current_total > peak_value:
                peak_value = current_total

            if self.risk_manager.check_max_drawdown(peak_value, current_total):
                logger.warning(f"Max drawdown limit reached on {date}")
                break

        if current_position is not None:
            final_price = df["close"].iloc[-1]
            commission = final_price * current_position.quantity * self.config.commission
            profit = (final_price - current_position.entry_price) * current_position.quantity - commission
            profit_pct = profit / (current_position.entry_price * current_position.quantity)

            trade = Trade(
                entry_date=current_position.entry_date,
                exit_date=df.index[-1],
                symbol=symbol,
                side=current_position.side,
                entry_price=current_position.entry_price,
                exit_price=final_price,
                quantity=current_position.quantity,
                commission=commission,
                slippage=0,
                profit=profit,
                profit_pct=profit_pct
            )
            trades_list.append(trade)

            portfolio.cash += final_price * current_position.quantity - commission

        result.final_value = portfolio.total_value
        result.total_return = (result.final_value - result.initial_cash) / result.initial_cash

        years = (result.end_date - result.start_date).days / 365.25
        if years > 0:
            result.annual_return = calculate_annual_return(result.total_return, years)

        if equity_values:
            equity_curve = pd.DataFrame({
                "date": equity_dates,
                "equity": equity_values
            }).set_index("date")
            result.equity_curve = equity_curve

            returns = equity_curve["equity"].pct_change().dropna()
            result.sharpe_ratio = calculate_sharpe_ratio(returns)

            result.max_drawdown = calculate_max_drawdown(equity_curve["equity"])

        result.trades = trades_list
        result.total_trades = len(trades_list)

        if trades_list:
            winning = [t for t in trades_list if t.profit > 0]
            losing = [t for t in trades_list if t.profit <= 0]
            result.winning_trades = len(winning)
            result.losing_trades = len(losing)
            result.win_rate = len(winning) / len(trades_list)

            if losing and winning:
                result.profit_loss_ratio = calculate_profit_loss_ratio(pd.DataFrame([t.to_dict() for t in trades_list]))

        if holding_days:
            result.avg_holding_days = sum(holding_days) / len(holding_days)

        logger.info(f"Backtest completed for {symbol}: Total Return={result.total_return:.2%}")

        return result

    def _create_empty_result(self, symbol: str, strategy_type: StrategyType) -> BacktestResult:
        return BacktestResult(
            strategy_name=strategy_type.value,
            symbol=symbol,
            start_date=datetime.now(),
            end_date=datetime.now(),
            initial_cash=self.config.initial_cash,
            final_value=self.config.initial_cash,
            total_return=0.0,
            annual_return=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_loss_ratio=0.0,
            sharpe_ratio=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            avg_holding_days=0.0
        )

    def backtest_multiple_strategies(
        self,
        df: pd.DataFrame,
        symbol: str,
        strategy_types: List[StrategyType] = None
    ) -> Dict[StrategyType, BacktestResult]:
        if strategy_types is None:
            strategy_types = list(StrategyType)

        results = {}
        for strategy_type in strategy_types:
            results[strategy_type] = self.backtest(df, symbol, strategy_type)

        return results


class BacktestReportGenerator:
    def __init__(self, output_dir: Path = REPORTS_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def generate_report(
        self,
        results: Dict[StrategyType, BacktestResult] | List[BacktestResult],
        symbol: str
    ) -> str:
        if not isinstance(results, list):
            results = list(results.values())

        if not results:
            return "No backtest results to report."

        lines = []
        lines.append("=" * 100)
        lines.append("策略回测报告")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"标的: {symbol}")
        lines.append("=" * 100)

        for result in results:
            lines.append(f"\n{'─' * 80}")
            lines.append(f"策略: {result.strategy_name}")
            lines.append(f"回测区间: {result.start_date.strftime('%Y-%m-%d')} 至 {result.end_date.strftime('%Y-%m-%d')}")
            lines.append("─" * 80)

            lines.append("\n【收益指标】")
            lines.append(f"  初始资金: {result.initial_cash:,.2f}")
            lines.append(f"  最终价值: {result.final_value:,.2f}")
            lines.append(f"  总收益率: {result.total_return:.2%}")
            lines.append(f"  年化收益率: {result.annual_return:.2%}")

            lines.append("\n【风险指标】")
            lines.append(f"  最大回撤: {result.max_drawdown:.2%}")
            lines.append(f"  夏普比率: {result.sharpe_ratio:.2f}")

            lines.append("\n【交易统计】")
            lines.append(f"  总交易次数: {result.total_trades}")
            lines.append(f"  盈利交易: {result.winning_trades}")
            lines.append(f"  亏损交易: {result.losing_trades}")
            lines.append(f"  胜率: {result.win_rate:.2%}")
            lines.append(f"  盈亏比: {result.profit_loss_ratio:.2f}")
            lines.append(f"  平均持仓天数: {result.avg_holding_days:.1f}")

            if result.trades:
                lines.append("\n【交易记录】")
                lines.append(f"{'日期':<12} {'方向':<6} {'入场价':<10} {'出场价':<10} {'数量':<10} {'盈利':<12} {'盈亏%':<10}")
                lines.append("-" * 80)

                for trade in result.trades[-10:]:
                    lines.append(
                        f"{trade.entry_date.strftime('%Y-%m-%d'):<12} "
                        f"{trade.side.value:<6} "
                        f"{trade.entry_price:<10.2f} "
                        f"{trade.exit_price:<10.2f} "
                        f"{trade.quantity:<10} "
                        f"{trade.profit:<12.2f} "
                        f"{trade.profit_pct:.2%}"
                    )

        lines.append("\n" + "=" * 100)
        lines.append("免责声明: 回测结果仅供参考，不代表实际收益。过往表现不预示未来回报。")
        lines.append("=" * 100)

        return "\n".join(lines)

    def save_report(
        self,
        results: Dict[StrategyType, BacktestResult] | List[BacktestResult],
        symbol: str,
        format: str = "txt"
    ) -> Path:
        if format == "txt":
            return self._save_text_report(results, symbol)
        elif format == "csv":
            return self._save_csv_report(results, symbol)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _save_text_report(
        self,
        results: Dict[StrategyType, BacktestResult] | List[BacktestResult],
        symbol: str
    ) -> Path:
        content = self.generate_report(results, symbol)
        filename = f"backtest_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Saved backtest report to {filepath}")
        return filepath

    def _save_csv_report(
        self,
        results: Dict[StrategyType, BacktestResult] | List[BacktestResult],
        symbol: str
    ) -> Path:
        if not isinstance(results, list):
            results = list(results.values())

        all_data = []
        for result in results:
            all_data.append(result.to_dict())

        if all_data:
            df = pd.DataFrame(all_data)
            filename = f"backtest_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            filepath = self.output_dir / filename
            save_to_csv(df, filepath)
            logger.info(f"Saved backtest CSV to {filepath}")
            return filepath

        return self.output_dir / "empty_backtest.csv"


class PortfolioBacktester:
    def __init__(self, config: BACKTEST_CONFIG = BACKTEST_CONFIG):
        self.config = config
        self.backtest_engine = BacktestEngine(config)

    def backtest_portfolio(
        self,
        stock_data: Dict[str, pd.DataFrame],
        strategy_type: StrategyType = StrategyType.COMBINED,
        initial_cash: float = None
    ) -> List[BacktestResult]:
        if initial_cash is None:
            initial_cash = self.config.initial_cash

        per_stock_cash = initial_cash / len(stock_data) if stock_data else initial_cash

        results = []
        for symbol, df in stock_data.items():
            result = self.backtest_engine.backtest(
                df, symbol, strategy_type, per_stock_cash
            )
            results.append(result)

        return results

    def generate_portfolio_summary(
        self,
        results: List[BacktestResult]
    ) -> Dict:
        if not results:
            return {}

        total_return = sum(r.total_return for r in results) / len(results)
        avg_sharpe = sum(r.sharpe_ratio for r in results) / len(results)
        avg_max_dd = sum(r.max_drawdown for r in results) / len(results)
        total_trades = sum(r.total_trades for r in results)

        return {
            "total_stocks": len(results),
            "avg_return": total_return,
            "avg_sharpe": avg_sharpe,
            "avg_max_drawdown": avg_max_dd,
            "total_trades": total_trades
        }


if __name__ == "__main__":
    import yfinance as yf

    print("Testing Backtest Module...")

    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="2y")

    if not df.empty:
        engine = BacktestEngine()

        result = engine.backtest(df, "AAPL", StrategyType.COMBINED)

        print(f"\nBacktest Results for AAPL:")
        print(f"  Total Return: {result.total_return:.2%}")
        print(f"  Annual Return: {result.annual_return:.2%}")
        print(f"  Max Drawdown: {result.max_drawdown:.2%}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  Win Rate: {result.win_rate:.2%}")
        print(f"  Total Trades: {result.total_trades}")

        report_gen = BacktestReportGenerator()
        report = report_gen.generate_report([result], "AAPL")
        print("\n" + report)
