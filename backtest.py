"""
backtest.py — Paper portfolio simulation and performance analytics.

Converts regime-conditioned factor weights into ETF allocations,
simulates a compounding dollar portfolio, and benchmarks against SPY.
"""

import numpy as np
import pandas as pd


FAMA_COLS = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']

# Fama-French factor → investable ETF proxy
FACTOR_ETF = {
    'Mkt-RF': 'SPY',    # broad market
    'SMB':    'IWM',    # small cap (Russell 2000)
    'HML':    'IVE',    # value (S&P 500 Value)
    'RMW':    'QUAL',   # quality / high profitability
    'CMA':    'USMV',   # low vol / conservative
    'Mom':    'MTUM',   # momentum
}

BENCHMARK = 'SPY'


class ETFMapper:
    """
    Converts numpy factor weight arrays (aligned to FAMA_COLS)
    into normalized ETF allocation dicts.
    """

    def __init__(self, factor_etf: dict = FACTOR_ETF):
        self.factor_etf = factor_etf

    def convert(self, weight_array: np.ndarray) -> dict:
        """
        Maps a factor weight array → {etf: weight}.
        Handles cases where multiple factors share the same ETF (e.g. SPY).
        Returns weights normalized to sum to 1.
        """
        etf_w = {}
        for factor, w in zip(FAMA_COLS, weight_array):
            etf = self.factor_etf.get(factor)
            if etf:
                etf_w[etf] = etf_w.get(etf, 0) + w
        total = sum(etf_w.values())
        return {k: v / total for k, v in etf_w.items()}

    def convert_all(self, regime_weights: dict) -> dict:
        """Convert all regime weight arrays at once. Returns {regime: {etf: weight}}."""
        return {r: self.convert(w) for r, w in regime_weights.items()}

    def print_allocations(self, regime_etf_weights: dict):
        print("\nETF allocation per regime:")
        for regime, etf_w in regime_etf_weights.items():
            allocs = {k: f'{v:.1%}' for k, v in etf_w.items()}
            print(f"  Regime {regime}: {allocs}")


class Portfolio:
    """
    Simulates a compounding dollar portfolio that rebalances daily
    based on regime-conditioned ETF weights.

    Parameters
    ----------
    starting_capital : initial portfolio value in dollars
    benchmark        : ticker used as the benchmark (default: SPY)
    """

    def __init__(self, starting_capital: float = 10_000, benchmark: str = BENCHMARK):
        self.starting_capital = starting_capital
        self.benchmark        = benchmark
        self.results: pd.DataFrame = pd.DataFrame()

    def run(
        self,
        etf_returns:        pd.DataFrame,
        regime_series:      pd.Series,
        regime_etf_weights: dict,
        train_cutoff:       str,
        regime_probs:       pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Simulate the portfolio day by day.

        Parameters
        ----------
        etf_returns         : daily returns DataFrame (dates × ETF tickers)
        regime_series       : hard regime label per date (lagged 1 day)
        regime_etf_weights  : {regime: {etf: weight}}
        train_cutoff        : date string splitting train/test periods
        regime_probs        : optional soft posterior probabilities (dates × regimes)
                              If provided, daily weights are a probability-weighted
                              blend of all regime weight vectors. This smooths out
                              abrupt rebalancing when the HMM is uncertain.
                              If None, hard regime assignment is used.

        Returns
        -------
        results DataFrame with columns: portfolio, benchmark, regime,
                                        port_return, bench_return, period
        """
        portfolio_nav = self.starting_capital
        benchmark_nav = self.starting_capital
        cutoff        = pd.Timestamp(train_cutoff)
        use_soft      = regime_probs is not None
        records       = []

        # Precompute ETF weight arrays for blending (aligned to regime ids)
        all_etfs     = list(set(FACTOR_ETF.values()))
        regime_arrays = {
            r: np.array([etf_w.get(etf, 0) for etf in all_etfs])
            for r, etf_w in regime_etf_weights.items()
        }

        for date, ret_row in etf_returns.iterrows():
            if use_soft and date in regime_probs.index:
                # Soft blend: weighted average of all regime weight vectors
                probs      = regime_probs.loc[date].values  # shape (n_regimes,)
                blended_w  = sum(
                    probs[r] * regime_arrays[r]
                    for r in regime_arrays
                )
                target_w = {etf: blended_w[i] for i, etf in enumerate(all_etfs)}
            else:
                # Hard assignment fallback
                target_w = regime_etf_weights[regime_series.loc[date]]

            port_return  = sum(
                target_w.get(etf, 0) * ret_row.get(etf, 0)
                for etf in FACTOR_ETF.values()
            )
            bench_return = ret_row.get(self.benchmark, 0)

            portfolio_nav *= (1 + port_return)
            benchmark_nav *= (1 + bench_return)

            records.append({
                'date':         date,
                'portfolio':    portfolio_nav,
                'benchmark':    benchmark_nav,
                'regime':       regime_series.loc[date],
                'port_return':  port_return,
                'bench_return': bench_return,
                'period':       'train' if date < cutoff else 'test',
            })

        self.results = pd.DataFrame(records).set_index('date')
        return self.results


class PerformanceAnalyzer:
    """Computes and displays performance metrics from backtest results."""

    TRADING_DAYS = 252

    def __init__(self, results: pd.DataFrame, starting_capital: float = 10_000):
        self.results          = results
        self.starting_capital = starting_capital

    def _metrics(self, returns: pd.Series, label: str) -> dict:
        ann_ret = returns.mean() * self.TRADING_DAYS
        ann_vol = returns.std()  * np.sqrt(self.TRADING_DAYS)
        sharpe  = ann_ret / ann_vol if ann_vol > 1e-10 else np.nan
        cum     = (1 + returns).cumprod()
        max_dd  = ((cum - cum.cummax()) / cum.cummax()).min()
        final   = self.starting_capital * cum.iloc[-1]
        return {
            'Strategy':     label,
            'Ann. Return':  f'{ann_ret:.2%}',
            'Ann. Vol':     f'{ann_vol:.2%}',
            'Sharpe':       f'{sharpe:.2f}',
            'Max Drawdown': f'{max_dd:.2%}',
            'Final Value':  f'${final:,.0f}',
        }

    def compute_all(self) -> pd.DataFrame:
        """Returns a DataFrame of metrics split by full / train / test period."""
        rows = []
        periods = {
            'Full Period':  self.results['period'].isin(['train', 'test']),
            'Train Period': self.results['period'] == 'train',
            'Test Period':  self.results['period'] == 'test',
        }
        for label, mask in periods.items():
            sub = self.results[mask]
            rows.append(self._metrics(sub['port_return'],  f'HMM Portfolio — {label}'))
            rows.append(self._metrics(sub['bench_return'], f'S&P 500       — {label}'))
        return pd.DataFrame(rows)

    def print_summary(self):
        metrics = self.compute_all()
        print('\n' + '='*75)
        current_period = None
        for _, row in metrics.iterrows():
            period = row['Strategy'].split('—')[-1].strip()
            if period != current_period:
                current_period = period
                print(f'\n── {period} ──')
                print(f"  {'':32} {'Ann.Ret':>9} {'Vol':>7} {'Sharpe':>8} "
                      f"{'MaxDD':>10} {'Final $':>12}")
            print(f"  {row['Strategy'][:32]:32} {row['Ann. Return']:>9} "
                  f"{row['Ann. Vol']:>7} {row['Sharpe']:>8} "
                  f"{row['Max Drawdown']:>10} {row['Final Value']:>12}")
        print('='*75)