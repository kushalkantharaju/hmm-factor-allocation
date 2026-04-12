"""
main.py — HMM Regime-Conditioned Factor Portfolio
==========================================
Run this file to execute the full pipeline end to end:
  1. Auto-detect whether a saved model exists and load it, or train one from scratch
  2. Build regime series with no lookahead bias
  3. Download ETF prices and simulate dollar portfolio
  4. Print performance metrics
  5. Plot NAV, regimes, and drawdown

Usage
-----
  python main.py            # auto-loads saved model, or trains one if none exists
  python main.py --rebuild  # force retrain even if a saved model exists
"""

import argparse
import os
import pandas as pd

from data     import MarketDataLoader, build_master
from regime   import RegimeModel, FAMA_COLS
from backtest import ETFMapper, Portfolio, PerformanceAnalyzer, FACTOR_ETF, BENCHMARK
from plotting import BacktestPlotter


# ── Configuration ─────────────────────────────────────────────────────────────

TRAIN_CUTOFF     = '2020-01-01'
STARTING_CAPITAL = 10_000
MODEL_PATH       = 'regime_model'      # prefix for saved model files
DATA_PATH        = 'data_cache'        # prefix for saved data files

# Default HMM parameters — only used when training from scratch
DEFAULT_N_REGIMES    = 3
DEFAULT_RANDOM_STATE = 21
DEFAULT_N_ITER       = 1000
DEFAULT_WEIGHT_METHOD = 'risk_parity'   # 'risk_parity' or 'max_sharpe'

# Regime label alignment: test regime ids → train regime ids
# Update this if you change DEFAULT_N_REGIMES
LABEL_MAP = {
    0: 2,   # Transitional
    1: 1,   # Crisis
    2: 0,   # Low Vol
}


# ── Data loading ──────────────────────────────────────────────────────────────

def get_data(force_rebuild: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (train, test, fama_df).
    Loads from cached pickles if they exist, otherwise downloads fresh data.
    """
    train_file = f'{DATA_PATH}_train.pkl'
    test_file  = f'{DATA_PATH}_test.pkl'
    fama_file  = f'{DATA_PATH}_fama.pkl'

    if not force_rebuild and all(os.path.exists(f) for f in [train_file, test_file, fama_file]):
        print("Found cached data — loading.")
        train   = pd.read_pickle(train_file)
        test    = pd.read_pickle(test_file)
        fama_df = pd.read_pickle(fama_file)
    else:
        print("No cached data found — downloading.")
        master, fama_df = build_master()
        train = master[master.index <  TRAIN_CUTOFF]
        test  = master[master.index >= TRAIN_CUTOFF]

        train.to_pickle(train_file)
        test.to_pickle(test_file)
        fama_df.to_pickle(fama_file)
        print(f"Data cached to: {DATA_PATH}_*.pkl")

    print(f"Train: {train.index[0].date()} → {train.index[-1].date()} ({len(train)} days)")
    print(f"Test:  {test.index[0].date()}  → {test.index[-1].date()} ({len(test)} days)")
    return train, test, fama_df


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(rm: RegimeModel, train: pd.DataFrame, test: pd.DataFrame):
    """Core backtest pipeline given a fitted RegimeModel. Returns results DataFrame."""

    # ── Step 1: Build lagged regime labels + soft probabilities (no lookahead)
    regime_series, regime_probs = rm.build_full_regime_series(
        train_features = train,
        test_features  = test,
        label_map      = LABEL_MAP,
    )

    inspect_cols = ['VIX Forward Looking',
                    'Realized Volatility(annualized 21 day)',
                    'Ten-Two Year Treasury Diff']

    print("\n── Train regime characteristics ──")
    train_regimes = rm.predict(train)
    print(train[inspect_cols].join(train_regimes).groupby('regime').mean())

    print("\n── Test regime characteristics ──")
    test_regimes = rm.predict(test)
    print(test[inspect_cols].join(test_regimes).groupby('regime').mean())

    # ── Step 2: Download ETF prices aligned to regime dates
    all_tickers = list(set(FACTOR_ETF.values()) | {BENCHMARK})
    start       = regime_series.index.min()
    end         = regime_series.index.max()

    print(f"\nDownloading ETFs: {all_tickers}")
    prices      = MarketDataLoader().fetch_etf_prices(all_tickers, str(start.date()), str(end.date()))
    etf_returns = prices.pct_change().dropna()

    # Align dates across all three
    common_dates  = regime_series.index.intersection(etf_returns.index)
    etf_returns   = etf_returns.loc[common_dates]
    regime_series = regime_series.loc[common_dates]
    regime_probs  = regime_probs.loc[common_dates]
    print(f"Trading days in backtest: {len(etf_returns)}")

    # ── Step 3: Convert factor weights → ETF weights
    mapper             = ETFMapper()
    regime_etf_weights = mapper.convert_all(rm.regime_weights)
    mapper.print_allocations(regime_etf_weights)

    # ── Step 4: Simulate portfolio with soft regime blending
    portfolio = Portfolio(starting_capital=STARTING_CAPITAL)
    results   = portfolio.run(
        etf_returns        = etf_returns,
        regime_series      = regime_series,
        regime_etf_weights = regime_etf_weights,
        train_cutoff       = TRAIN_CUTOFF,
        regime_probs       = regime_probs,   # soft blending enabled
    )

    # ── Step 5: Performance metrics
    PerformanceAnalyzer(results, STARTING_CAPITAL).print_summary()

    # ── Step 6: Plot
    BacktestPlotter(results, TRAIN_CUTOFF).plot(save_path='hmm_backtest.png', show=True)

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild', action='store_true',
                        help='Force retrain the model and re-download data even if saved files exist')
    args = parser.parse_args()

    # Get data (cached or fresh)
    train, test, fama_df = get_data(force_rebuild=args.rebuild)

    # Get model — loads from disk if it exists, trains from scratch if not
    # --rebuild deletes any saved model first so load_or_create is forced to retrain
    if args.rebuild and os.path.exists(f'{MODEL_PATH}_hmm.pkl'):
        print("--rebuild flag set — removing saved model to force retrain.")
        for suffix in ['_hmm.pkl', '_scaler.pkl', '_weights.pkl', '_stats.pkl']:
            path = f'{MODEL_PATH}{suffix}'
            if os.path.exists(path):
                os.remove(path)

    rm = RegimeModel.load_or_create(
        train_features = train,
        fama_df        = fama_df[FAMA_COLS].reindex(train.index).dropna(),
        path           = MODEL_PATH,
        n_regimes      = DEFAULT_N_REGIMES,
        random_state   = DEFAULT_RANDOM_STATE,
        n_iter         = DEFAULT_N_ITER,
        weight_method  = DEFAULT_WEIGHT_METHOD,
    )

    rm.print_weights()

    # Run backtest
    results = run_pipeline(rm, train, test)
    results.to_pickle('backtest_results.pkl')
    print("\nSaved: backtest_results.pkl")