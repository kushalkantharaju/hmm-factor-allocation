"""
plotting.py — Visualization for HMM regime-conditioned backtest results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec


REGIME_COLORS = {
    0: '#3B82F6',   # blue  — Low Vol
    1: '#EF4444',   # red   — Crisis
    2: '#10B981',   # green — Transitional
    3: '#F59E0B',   # amber
    4: '#8B5CF6',   # purple
}

REGIME_LABELS = {
    0: 'Low Vol',
    1: 'Crisis',
    2: 'Transitional',
}

DARK_BG    = '#0F1117'
GRID_COLOR = '#1F2937'
TEXT_COLOR = '#9CA3AF'
SPLIT_COLOR = '#FBBF24'


class BacktestPlotter:
    """
    Produces a 3-panel chart:
      Panel 1 — NAV curves (portfolio vs benchmark)
      Panel 2 — Regime bands over time
      Panel 3 — Drawdown comparison
    """

    def __init__(self, results: pd.DataFrame, train_cutoff: str):
        self.results      = results
        self.train_cutoff = pd.Timestamp(train_cutoff)

    def _style_ax(self, ax):
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.spines[:].set_color(GRID_COLOR)
        ax.grid(axis='y', color=GRID_COLOR, linewidth=0.5, alpha=0.6)

    def _add_split_line(self, ax):
        ax.axvline(self.train_cutoff, color=SPLIT_COLOR,
                   linewidth=1, linestyle=':', alpha=0.9, zorder=4)

    def plot(self, save_path: str = 'hmm_backtest.png', show: bool = True):
        fig = plt.figure(figsize=(16, 11), facecolor=DARK_BG)
        gs  = GridSpec(3, 1, figure=fig, height_ratios=[3, 1, 1], hspace=0.06)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax3 = fig.add_subplot(gs[2], sharex=ax1)

        for ax in [ax1, ax2, ax3]:
            self._style_ax(ax)

        self._plot_nav(ax1)
        self._plot_regimes(ax2)
        self._plot_drawdown(ax3)

        self._format_xaxis(ax3)
        plt.setp(ax1.xaxis.get_majorticklabels(), visible=False)
        plt.setp(ax2.xaxis.get_majorticklabels(), visible=False)

        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=DARK_BG, edgecolor='none')
        print(f"Saved: {save_path}")
        if show:
            plt.show()

    def _plot_nav(self, ax):
        ax.plot(self.results.index, self.results['portfolio'],
                color='#60A5FA', linewidth=1.8, label='HMM Factor Portfolio', zorder=3)
        ax.plot(self.results.index, self.results['benchmark'],
                color=TEXT_COLOR, linewidth=1.2, linestyle='--',
                label='S&P 500 (SPY)', zorder=2)
        self._add_split_line(ax)
        ax.text(self.train_cutoff, self.results['portfolio'].max() * 0.96,
                '  OOS →', color=SPLIT_COLOR, fontsize=8, va='top')
        ax.set_ylabel('Portfolio Value ($)', color=TEXT_COLOR, fontsize=10)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'${x:,.0f}')
        )
        ax.legend(facecolor=GRID_COLOR, edgecolor='#374151',
                  labelcolor='white', fontsize=9)
        ax.set_title(
            'HMM Regime-Conditioned Factor Portfolio vs S&P 500\n$10,000 Starting Capital',
            color='white', fontsize=13, pad=10, fontweight='bold',
        )

    def _plot_regimes(self, ax):
        unique_regimes = sorted(self.results['regime'].unique())
        for regime in unique_regimes:
            mask  = self.results['regime'] == regime
            label = f"Regime {regime} ({REGIME_LABELS.get(regime, '?')})"
            ax.fill_between(
                self.results.index, 0, 1, where=mask,
                color=REGIME_COLORS.get(regime, '#6B7280'),
                alpha=0.85, label=label,
            )
        self._add_split_line(ax)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel('Regime', color=TEXT_COLOR, fontsize=9)
        ax.legend(facecolor=GRID_COLOR, edgecolor='#374151',
                  labelcolor='white', fontsize=8,
                  loc='upper right', ncol=len(unique_regimes))

    def _plot_drawdown(self, ax):
        port_cum  = (1 + self.results['port_return']).cumprod()
        bench_cum = (1 + self.results['bench_return']).cumprod()
        port_dd   = (port_cum  - port_cum.cummax())  / port_cum.cummax()
        bench_dd  = (bench_cum - bench_cum.cummax()) / bench_cum.cummax()

        ax.fill_between(self.results.index, port_dd,  0,
                        color='#3B82F6', alpha=0.5, label='Portfolio')
        ax.fill_between(self.results.index, bench_dd, 0,
                        color=TEXT_COLOR, alpha=0.3, label='S&P 500')
        self._add_split_line(ax)
        ax.set_ylabel('Drawdown', color=TEXT_COLOR, fontsize=9)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{x:.0%}')
        )
        ax.legend(facecolor=GRID_COLOR, edgecolor='#374151',
                  labelcolor='white', fontsize=8)

    def _format_xaxis(self, ax):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), color=TEXT_COLOR)


        
