"""
data.py — Market data ingestion
Handles FRED macro series, Fama-French factors, and market price data.
"""

import io
import zipfile
import requests
import pandas as pd
import yfinance as yf


FRED_API_KEY = 'de8cb32049ae576e215c319562e439ba'

FRED_SERIES = {
    'Ten-Two Year Treasury Diff': 'T10Y2Y',
    'Bond to Ten Year Treasury':  'BAA10Y',
}

FF_FACTORS = {
    'five': 'F-F_Research_Data_5_Factors_2x3_daily',
    'mom':  'F-F_Momentum_Factor_daily',
}


class FREDLoader:
    """Fetches macro time series from the St. Louis FRED API."""

    def __init__(self, api_key: str = FRED_API_KEY):
        self.api_key = api_key

    def fetch_series(
        self,
        series_id: str,
        start: str = '2000-01-01',
        end:   str = '2026-01-31',
        name:  str = None,
    ) -> pd.Series:
        url = (
            f'https://api.stlouisfed.org/fred/series/observations'
            f'?series_id={series_id}&api_key={self.api_key}'
            f'&file_type=json&observation_start={start}&observation_end={end}'
        )
        obs = requests.get(url).json()['observations']
        dates  = [o['date']  for o in obs]
        values = [o['value'] for o in obs]
        series = pd.to_numeric(
            pd.Series(values, index=pd.to_datetime(dates), name=name or series_id),
            errors='coerce',
        )
        series.sort_index(inplace=True)
        return series

    def fetch_all(self, start: str = '2000-01-01', end: str = '2026-01-31') -> pd.DataFrame:
        """Returns a DataFrame with all configured FRED series."""
        return pd.concat(
            [self.fetch_series(sid, start, end, name=name)
             for name, sid in FRED_SERIES.items()],
            axis=1,
        )


class FamaFrenchLoader:
    """Downloads and parses Fama-French factor data from Ken French's website."""

    BASE_URL = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{}_CSV.zip'

    def _download(self, factor_name: str, skiprows: int) -> pd.DataFrame:
        url = self.BASE_URL.format(factor_name)
        zf  = zipfile.ZipFile(io.BytesIO(requests.get(url).content))
        df  = pd.read_csv(zf.open(zf.namelist()[0]), skiprows=skiprows, index_col=0)
        df  = df[df.index.astype(str).str.strip().str.len() == 8]
        df.index = pd.to_datetime(df.index.astype(str).str.strip(), format='%Y%m%d')
        df  = df.apply(pd.to_numeric, errors='coerce') / 100
        df.dropna(how='all', inplace=True)
        return df

    def fetch_five_factors(self) -> pd.DataFrame:
        return self._download(FF_FACTORS['five'], skiprows=3)

    def fetch_momentum(self) -> pd.DataFrame:
        return self._download(FF_FACTORS['mom'], skiprows=12)

    def fetch_all(self, start: str = '2000-01-01', end: str = '2026-01-31') -> pd.DataFrame:
        ff5 = self.fetch_five_factors()
        mom = self.fetch_momentum()
        df  = pd.concat([ff5, mom], axis=1)
        return df[(df.index >= start) & (df.index <= end)]


class MarketDataLoader:
    """Downloads SPY and VIX price data and engineers features."""

    def fetch_spy_features(self, start: str = '2000-01-01', end: str = '2026-01-31') -> pd.DataFrame:
        raw     = yf.download('SPY', start=start, end=end, auto_adjust=True, progress=False)
        close   = raw['Close'].squeeze()
        returns = close.pct_change()
        return pd.DataFrame({
            'Realized Volatility(annualized 21 day)': returns.rolling(21).std() * (252 ** 0.5),
            'Trailing Return(21 day)':                close.pct_change(21),
        })

    def fetch_vix_features(self, start: str = '2000-01-01', end: str = '2026-01-31') -> pd.DataFrame:
        raw   = yf.download('^VIX', start=start, end=end, auto_adjust=True, progress=False)
        close = raw['Close'].squeeze()
        return pd.DataFrame({
            'VIX Forward Looking':  close,
            'VIX change 21 day':    close.pct_change(21),
        })

    def fetch_etf_prices(
        self,
        tickers:    list,
        start:      str,
        end:        str,
    ) -> pd.DataFrame:
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)['Close']
        return raw.dropna()


def build_master(
    start: str = '2000-01-01',
    end:   str = '2026-01-31',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds and returns (clean_master, fama_df).
    clean_master: HMM features only (for regime fitting)
    fama_df:      Fama-French factor returns (for portfolio construction)
    """
    fred   = FREDLoader().fetch_all(start, end)
    spy    = MarketDataLoader().fetch_spy_features(start, end)
    vix    = MarketDataLoader().fetch_vix_features(start, end)
    fama   = FamaFrenchLoader().fetch_all(start, end)

    master = pd.concat([vix, spy, fred], axis=1).dropna()
    fama   = fama.reindex(master.index).dropna()
    master = master.reindex(fama.index)

    return master, fama
