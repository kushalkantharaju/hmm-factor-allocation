"""
regime.py — HMM regime detection and portfolio weight optimization.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import minimize

DEFAULT_SAVE_PATH = 'regime_model'


HMM_FEATURES = [
    'VIX Forward Looking',
    'VIX change 21 day',
    'Realized Volatility(annualized 21 day)',
    'Trailing Return(21 day)',
    'Ten-Two Year Treasury Diff',
    'Bond to Ten Year Treasury',
]

FAMA_COLS = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']

REGIME_LABELS = {
    0: 'Low Vol',
    1: 'Crisis',
    2: 'Transitional',
}


class RegimeModel:
    """
    Fits a Gaussian HMM on macro features to identify market regimes,
    then optimizes max-Sharpe factor weights per regime.

    Parameters
    ----------
    n_regimes   : number of hidden states
    random_state: random seed for reproducibility
    n_iter      : HMM training iterations
    """

    def __init__(self, n_regimes: int = 3, random_state: int = 21, n_iter: int = 1000):
        self.n_regimes    = n_regimes
        self.random_state = random_state
        self.n_iter       = n_iter

        self.scaler   = StandardScaler()
        self.model    = GaussianHMM(
            n_components    = n_regimes,
            covariance_type = 'full',
            n_iter          = n_iter,
            random_state    = random_state,
        )

        # Set after fitting
        self.regime_stats:   dict = {}   # {regime: {'mean': arr, 'cov': arr}}
        self.regime_weights: dict = {}   # {regime: np.array aligned to fama_cols}
        self.label_map:      dict = {}   # maps test regime ids → train regime ids

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, features_df: pd.DataFrame) -> 'RegimeModel':
        """
        Fit scaler + HMM on the provided feature DataFrame.
        Expects columns matching HMM_FEATURES.
        """
        X = features_df[HMM_FEATURES].values
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        return self

    def predict(self, features_df: pd.DataFrame) -> pd.Series:
        """Predict hard regime labels for a feature DataFrame."""
        X_scaled = self.scaler.transform(features_df[HMM_FEATURES].values)
        labels   = self.model.predict(X_scaled)
        return pd.Series(labels, index=features_df.index, name='regime')

    def predict_proba(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the posterior probability of each regime for every date.
        Shape: (n_dates, n_regimes)
        Used for soft blending — today's weights are a probability-weighted
        mix of all regime weight vectors rather than a hard assignment.
        """
        X_scaled = self.scaler.transform(features_df[HMM_FEATURES].values)
        probs    = self.model.predict_proba(X_scaled)
        cols     = [f'regime_{r}' for r in range(self.n_regimes)]
        return pd.DataFrame(probs, index=features_df.index, columns=cols)

    # ── Weight optimization ───────────────────────────────────────────────────

    def compute_regime_stats(self, master_df: pd.DataFrame, regime_series: pd.Series) -> 'RegimeModel':
        """
        Computes per-regime mean and covariance of Fama-French factor returns.
        master_df must contain FAMA_COLS.
        """
        for regime in range(self.n_regimes):
            mask   = regime_series == regime
            subset = master_df.loc[mask, FAMA_COLS]
            self.regime_stats[regime] = {
                'mean': subset.mean().values,
                'cov':  subset.cov().values,
            }
        return self

    @staticmethod
    def _max_sharpe_weights(mean_returns: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Solves for the long-only max-Sharpe portfolio weights.
        Sensitive to mean return estimates — use with caution on small regime samples.
        """
        n = len(mean_returns)

        def neg_sharpe(w):
            ret = w @ mean_returns
            vol = np.sqrt(w @ cov_matrix @ w)
            return -ret / vol if vol > 1e-10 else 1e10

        result = minimize(
            neg_sharpe,
            x0          = np.ones(n) / n,
            bounds      = [(0, 1)] * n,
            constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1},
        )
        return result.x

    @staticmethod
    def _risk_parity_weights(cov_matrix: np.ndarray) -> np.ndarray:
        """
        Computes risk parity weights — each factor contributes equally
        to total portfolio volatility.

        Unlike max-Sharpe, this only uses the covariance matrix (not mean
        returns), making it far more stable across regime samples since
        covariance estimates are much less noisy than return estimates.
        """
        n = len(cov_matrix)

        def risk_concentration(w):
            w            = np.abs(w)
            port_vol     = np.sqrt(w @ cov_matrix @ w)
            marginal     = cov_matrix @ w
            risk_contrib = w * marginal / port_vol
            target       = port_vol / n
            return np.sum((risk_contrib - target) ** 2)

        result = minimize(
            risk_concentration,
            x0          = np.ones(n) / n,
            bounds      = [(1e-6, 1)] * n,
            constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1},
            method      = 'SLSQP',
        )
        w = np.abs(result.x)
        return w / w.sum()

    def optimize_weights(self, method: str = 'risk_parity') -> 'RegimeModel':
        """
        Compute factor weights for each regime.

        Parameters
        ----------
        method : 'risk_parity' (default) or 'max_sharpe'
            risk_parity — equal risk contribution, uses only covariance.
                          More robust to small/noisy regime samples.
            max_sharpe  — maximizes Sharpe ratio, uses mean + covariance.
                          More sensitive to noisy return estimates.
        """
        if not self.regime_stats:
            raise RuntimeError("Call compute_regime_stats() before optimize_weights().")

        for regime, stats in self.regime_stats.items():
            if method == 'risk_parity':
                self.regime_weights[regime] = self._risk_parity_weights(stats['cov'])
            elif method == 'max_sharpe':
                self.regime_weights[regime] = self._max_sharpe_weights(stats['mean'], stats['cov'])
            else:
                raise ValueError(f"Unknown method '{method}'. Use 'risk_parity' or 'max_sharpe'.")

        self._weight_method = method
        return self

    # ── Persistence ───────────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        """True if the model has been trained and weights computed."""
        return bool(self.regime_weights) and hasattr(self.model, 'transmat_')

    def save(self, path: str = DEFAULT_SAVE_PATH):
        """
        Saves the full model state to disk.
        Creates four files: {path}_hmm.pkl, _scaler.pkl, _weights.pkl, _stats.pkl
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() and optimize_weights() first.")
        joblib.dump(self.model,          f'{path}_hmm.pkl')
        joblib.dump(self.scaler,         f'{path}_scaler.pkl')
        joblib.dump(self.regime_weights, f'{path}_weights.pkl')
        joblib.dump(self.regime_stats,   f'{path}_stats.pkl')
        print(f"Model saved to: {path}_*.pkl")

    @classmethod
    def load(cls, path: str = DEFAULT_SAVE_PATH) -> 'RegimeModel':
        """
        Loads a previously saved model from disk.
        Raises FileNotFoundError if the files don't exist.
        """
        required = [f'{path}_hmm.pkl', f'{path}_scaler.pkl',
                    f'{path}_weights.pkl', f'{path}_stats.pkl']
        missing = [f for f in required if not os.path.exists(f)]
        if missing:
            raise FileNotFoundError(f"Missing model files: {missing}")

        rm                = cls.__new__(cls)   # bypass __init__ to avoid creating a blank HMM
        rm.model          = joblib.load(f'{path}_hmm.pkl')
        rm.scaler         = joblib.load(f'{path}_scaler.pkl')
        rm.regime_weights = joblib.load(f'{path}_weights.pkl')
        rm.regime_stats   = joblib.load(f'{path}_stats.pkl')
        rm.n_regimes      = rm.model.n_components
        rm.random_state   = rm.model.random_state
        rm.n_iter         = rm.model.n_iter
        rm.label_map      = {}
        print(f"Model loaded from: {path}_*.pkl  ({rm.n_regimes} regimes)")
        return rm

    @classmethod
    def load_or_create(
        cls,
        train_features: pd.DataFrame,
        fama_df:        pd.DataFrame,
        path:           str  = DEFAULT_SAVE_PATH,
        n_regimes:      int  = 3,
        random_state:   int  = 21,
        n_iter:         int  = 1000,
        weight_method:  str  = 'risk_parity',
    ) -> 'RegimeModel':
        """
        Loads an existing model from disk if one is found.
        If no saved model exists, trains a new one from scratch using
        default parameters and saves it for next time.

        Parameters
        ----------
        train_features : DataFrame with HMM_FEATURES columns (training period only)
        fama_df        : DataFrame with FAMA_COLS (training period only)
        path           : file path prefix for saved model files
        n_regimes      : number of hidden states (used only when creating)
        random_state   : random seed (used only when creating)
        n_iter         : HMM iterations (used only when creating)
        weight_method  : 'risk_parity' or 'max_sharpe' (used only when creating)
        """
        hmm_file = f'{path}_hmm.pkl'

        if os.path.exists(hmm_file):
            print(f"Found saved model at '{hmm_file}' — loading.")
            return cls.load(path)

        print(f"No saved model found at '{hmm_file}' — training from scratch.")
        rm = cls(n_regimes=n_regimes, random_state=random_state, n_iter=n_iter)
        rm.fit(train_features)

        train_regimes = rm.predict(train_features)
        fama_aligned  = fama_df.reindex(train_features.index).dropna()
        rm.compute_regime_stats(fama_aligned, train_regimes)
        rm.optimize_weights(method=weight_method)
        rm.save(path)
        return rm

    def print_weights(self):
        print("\nMax-Sharpe weights per regime:")
        for regime, weights in self.regime_weights.items():
            label = REGIME_LABELS.get(regime, f'Regime {regime}')
            print(f"\n  {label} (Regime {regime}):")
            for factor, w in zip(FAMA_COLS, weights):
                print(f"    {factor}: {w:.4f}")

    # ── Train/test regime alignment ───────────────────────────────────────────

    def build_full_regime_series(
        self,
        train_features: pd.DataFrame,
        test_features:  pd.DataFrame,
        label_map:      dict,
    ) -> tuple[pd.Series, pd.DataFrame]:
        """
        Predicts regimes on train and test separately, applies label_map
        to the test period, then combines and lags by 1 day.

        Returns
        -------
        regime_labels : hard regime assignment per date (lagged 1 day)
        regime_probs  : soft posterior probabilities per date (lagged 1 day)
                        shape (n_dates, n_regimes), columns remapped via label_map
        """
        self.label_map = label_map

        # Hard labels
        train_labels = self.predict(train_features)
        test_labels  = self.predict(test_features).map(label_map)
        combined_labels = pd.concat([train_labels, test_labels]).sort_index()
        lagged_labels   = combined_labels.shift(1).dropna()

        # Soft probabilities — remap columns to match label_map
        train_probs = self.predict_proba(train_features)
        test_probs  = self.predict_proba(test_features)

        # Reorder test probability columns so they align with remapped regime ids
        n = self.n_regimes
        col_map = {f'regime_{old}': f'regime_{new}' for old, new in label_map.items()}
        test_probs = test_probs.rename(columns=col_map)[train_probs.columns]

        combined_probs = pd.concat([train_probs, test_probs]).sort_index()
        lagged_probs   = combined_probs.shift(1).dropna()
        lagged_probs   = lagged_probs.loc[lagged_labels.index]

        print(f"Regime series: {lagged_labels.index[0].date()} → {lagged_labels.index[-1].date()}")
        print(f"Distribution:\n{lagged_labels.value_counts().sort_index().to_string()}")
        return lagged_labels, lagged_probs