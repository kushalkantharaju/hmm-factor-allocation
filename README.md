# hmm-factor-allocation
Macro regime detection via Hidden Markov Models, used to dynamically allocate across Fama-French factors.

Overview:
This project builds a regime-conditioned factor allocation system. A Gaussian Hidden Markov Model is trained on macro and volatility features to identify latent market regimes. Factor exposures — drawn from the Fama-French Five-Factor model — are then allocated based on the predicted regime, aiming to improve risk-adjusted returns over a static factor strategy.

Pipeline:
Data ingestion
→
Feature engineering
→
HMM training
→
Regime labeling
→
Factor allocation
→
Backtesting


Features: 
HMM regime detection — Gaussian HMM (via hmmlearn) trained on macro/vol features to identify latent states
Fama-French Five Factors — MKT-RF, SMB, HML, RMW, CMA sourced directly from Kenneth French's data library
Macro features — FRED-sourced indicators (yield curve, credit spreads, VIX, etc.) via direct API calls
Regime-conditioned allocation — factor weights shift dynamically based on the decoded regime sequence
Backtesting framework — performance attribution and comparison against static equal-weight factor baseline
Data sources
Fama-French factors — raw .zip downloads from mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
Macro indicators — FRED API using observation_start / observation_end parameters
Note: pandas-datareader's Fama-French reader is broken on Python 3.13 — this project uses direct HTTP downloads instead.

Requirements:
hmmlearn
numpy
pandas
matplotlib
scipy
requests
fredapi
jupyter

Status:
Active development. Feature engineering complete, HMM training, and regime labeling complete. Live trading implementation in progress.
