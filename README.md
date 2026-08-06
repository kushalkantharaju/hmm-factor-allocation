# hmm-factor-allocation

> Macro regime detection via Hidden Markov Models, used to dynamically allocate across Fama-French factors.

## Overview
This project builds a regime-conditioned factor allocation system. A Gaussian Hidden Markov Model is trained on macro and volatility features to identify latent market regimes. Factor exposures, drawn from the Fama-French Five-Factor model, are then allocated based on the predicted regime. This aims to improve risk-adjusted returns over a static factor strategy.

---

## Pipeline
1. Data ingestion
2. Feature engineering
3. HMM training
4. Regime labeling
5. Factor allocation
6. Backtesting

---

## Features
* **HMM regime detection**: Uses a Gaussian HMM (via `hmmlearn`) trained on macro and volatility features to identify latent states.
* **Fama-French Five Factors**: Incorporates MKT-RF, SMB, HML, RMW, and CMA, sourced directly from Kenneth French's data library.
* **Macro features**: Utilizes FRED-sourced indicators (such as the yield curve, credit spreads, and VIX) via direct API calls.
* **Regime-conditioned allocation**: Factor weights shift dynamically based on the decoded regime sequence.
* **Backtesting framework**: Features performance attribution and comparison against a static equal-weight factor baseline.

---

## Data Sources
* **Fama-French factors**: Downloaded as raw `.zip` files from `mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html`.
* **Macro indicators**: Sourced via the FRED API using `observation_start` and `observation_end` parameters.
* **Implementation Note**: This project uses direct HTTP downloads because `pandas-datareader`'s Fama-French reader is currently broken on Python 3.13.

---

## Usage
Run the main file (`main.py`) to execute the full pipeline end-to-end. Execution will auto-detect and load a saved model (or train a new one), build a regime series with no lookahead bias, download ETF prices, simulate a dollar portfolio, and generate performance metrics and plots.

**Standard Execution:**
```bash
python main.py
```
This command auto-loads a saved model, or trains one if none exists.

**Force Retrain:**
```bash
python main.py --rebuild
```
This flag forces the system to retrain the model even if a saved model already exists.

---

## Requirements
`hmmlearn`
`numpy`
`pandas`
`matplotlib`
`scipy`
`requests`
`fredapi`
`jupyter`

---

## Status
Feature engineering, HMM training, and regime labeling are complete. A live trading implementation is currently in progress.
