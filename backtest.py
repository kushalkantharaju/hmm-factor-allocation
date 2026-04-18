"""
backtest.py — Paper portfolio simulation and performance analytics.
 
Hybrid factor portfolio approach:
- Mkt-RF weight → direct SPY exposure (captures full market return)
- All other factor weights → long-short ETF spread tilts on top
- This mirrors how factor investing works in practice: hold the market,
  tilt around it using factor signals rather than running pure long-short.
"""
 
import numpy as np
import pandas as pd
 
 
FAMA_COLS = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'Mom']
 
# Mkt-RF: long only SPY (full market return)
# Others: long-short spreads applied as tilts on top of market exposure
FACTOR_PROXIES = {
    'Mkt-RF': ('SPY',  None),    # long only — full market return
    'SMB':    ('IWM',  'SPY'),   # small cap minus large cap spread
    'HML':    ('VLUE', 'VUG'),   # value minus growth spread
    'RMW':    ('QUAL', 'SPHB'),  # quality minus high beta spread
    'Mom':    ('MTUM', 'SPY'),   # momentum minus market spread
}
 
BENCHMARK = 'SPY'
 
ALL_TICKERS = list(set(
    t for long, short in FACTOR_PROXIES.values()
    for t in [long, short] if t is not None
) | {BENCHMARK})
 
 
class FactorReturnBuilder:
    """
    Computes hybrid factor returns:
 
    Mkt-RF → raw SPY return (full market exposure preserved)
    Others → long_etf_return - short_etf_return (spread tilts)
 
    The key insight: Mkt-RF captures broad upward market drift.
    Factor spreads capture incremental premia above market return.
    Combined via regime weights, the portfolio gets:
        - core market return from Mkt-RF allocation
        - factor tilt returns from spread allocations
    """
 
    def __init__(self, factor_proxies: dict = FACTOR_PROXIES):
        self.factor_proxies = factor_proxies
 
    def compute(self, etf_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame of factor returns (dates × factors).
        Mkt-RF = raw SPY return.
        All others = long ETF return - short ETF return.
        """
        factor_rets = {}
        for factor, (long_etf, short_etf) in self.factor_proxies.items():
            long_ret = etf_returns[long_etf]
            if short_etf is None:
                # Mkt-RF: keep full market return, no subtraction
                factor_rets[factor] = long_ret
            else:
                # Factor tilt: spread between long and short
                factor_rets[factor] = long_ret - etf_returns[short_etf]
        return pd.DataFrame(factor_rets, index=etf_returns.index)
 
    def print_proxies(self):
        print("\nFactor proxy construction:")
        for factor, (long_etf, short_etf) in self.factor_proxies.items():
            short_str = f'− {short_etf}' if short_etf else '(long only, full return)'
            print(f"  {factor:<8} = {long_etf} {short_str}")
 
 
class Portfolio:
    """
    Simulates a compounding dollar portfolio using hybrid factor returns.
 
    Daily portfolio return:
        = w_MktRF * SPY_return
        + w_SMB   * (IWM - SPY)
        + w_HML   * (VLUE - VUG)
        + w_RMW   * (QUAL - SPHB)
        + w_Mom   * (MTUM - SPY)
 
    The Mkt-RF weight ensures the portfolio has meaningful market
    exposure and captures upward market drift over time.
    """
 
    def __init__(self, starting_capital: float = 10_000, benchmark: str = BENCHMARK):
        self.starting_capital = starting_capital
        self.benchmark        = benchmark
        self.results: pd.DataFrame = pd.DataFrame()
 
    def run(
        self,
        factor_returns: pd.DataFrame,
        etf_returns:    pd.DataFrame,
        regime_series:  pd.Series,
        regime_weights: dict,
        train_cutoff:   str,
        regime_probs:   pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Simulate the portfolio day by day.
 
        Parameters
        ----------
        factor_returns  : hybrid factor returns from FactorReturnBuilder.compute()
        etf_returns     : raw ETF returns — used for benchmark only
        regime_series   : hard regime label per date (lagged 1 day)
        regime_weights  : {regime: np.array aligned to FAMA_COLS}
        train_cutoff    : date string splitting train/test periods
        regime_probs    : optional soft posterior probabilities (dates × regimes)
                          Enables smooth weight blending across regime uncertainty.
        """
        portfolio_nav = self.starting_capital
        benchmark_nav = self.starting_capital
        cutoff        = pd.Timestamp(train_cutoff)
        use_soft      = regime_probs is not None
        records       = []
 
        regime_arrays = {r: w for r, w in regime_weights.items()}
 
        for date in factor_returns.index:
            if date not in regime_series.index:
                continue
 
            f_row  = factor_returns.loc[date].values
            regime = regime_series.loc[date]
 
            if use_soft and date in regime_probs.index:
                probs   = regime_probs.loc[date].values
                weights = sum(probs[r] * regime_arrays[r] for r in regime_arrays)
            else:
                weights = regime_arrays[regime]
 
            port_return  = float(weights @ f_row)
            bench_return = float(etf_returns.loc[date].get(self.benchmark, 0))
 
            portfolio_nav *= (1 + port_return)
            benchmark_nav *= (1 + bench_return)
 
            records.append({
                'date':         date,
                'portfolio':    portfolio_nav,
                'benchmark':    benchmark_nav,
                'regime':       regime,
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
        rows    = []
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