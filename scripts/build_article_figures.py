"""Rebuild publication figures exclusively from the saved experiment outputs."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/figures'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
                     'legend.fontsize': 8.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 220})
COLORS = ['#777777', '#2563a6', '#d9822b', '#26846c', '#8a4a9b']
NAMES = ['equal_weight', 'baseline', 'pipeline1', 'pipeline2', 'pipeline3']
LABELS = ['Equal weight', 'Baseline GMV', 'Pipeline 1', 'Pipeline 2', 'Pipeline 3']

def save(fig, name):
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight', facecolor='white')
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)

coverage = pd.read_csv(ROOT / 'artifacts/tables/fundamental_coverage_by_year.csv')
fig, ax = plt.subplots(figsize=(6.8, 2.8), layout='constrained')
ax.plot(coverage.date, coverage.quality_coverage, color=COLORS[3], marker='o', markersize=4)
ax.axvspan(2017, 2021, color='#e1e8ef', alpha=.55)
ax.axvspan(2021, 2026.4, color='#e8f1e8', alpha=.6)
ax.text(2018.8, .57, 'Validation', ha='center', fontsize=9)
ax.text(2023.5, .57, 'Test', ha='center', fontsize=9)
ax.set(xlabel='Calendar year', ylabel='Available ratio entries', ylim=(.53, .88), xlim=(2009.7, 2026.4))
ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.grid(axis='y', alpha=.2)
save(fig, 'quality_coverage')

fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.6), sharex=True, layout='constrained')
for name, label, color in zip(NAMES, LABELS, COLORS):
    df = pd.read_csv(ROOT / f'artifacts/results/test/{name}/daily_returns.csv', parse_dates=['date']).set_index('date')
    wealth = (1 + df.net_return).cumprod()
    dd = wealth / wealth.cummax().clip(lower=1) - 1
    style = '--' if name == 'pipeline3' else '-'
    axes[0].plot(wealth.index, wealth, color=color, label=label, lw=1.25, ls=style)
    axes[1].plot(dd.index, dd, color=color, lw=1.1, ls=style)
axes[0].set(ylabel='Net wealth', title='Growth of unit initial capital')
axes[0].legend(ncol=3, loc='upper left', frameon=False)
axes[1].set(ylabel='Drawdown', xlabel='Test date')
axes[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
for ax in axes: ax.grid(axis='y', alpha=.2)
save(fig, 'test_performance')

pred = pd.read_csv(ROOT / 'artifacts/tables/pipeline3_test_predictions.csv').dropna(subset=['future_downside_variance'])
pred['quintile'] = pred.groupby('date').predicted_downside_variance.transform(lambda x: pd.qcut(x.rank(method='first'), 5, labels=False) + 1)
# Equal weighting of formation dates, matching the notebook diagnostic.
group = pred.groupby(['date', 'quintile'])[['predicted_downside_variance', 'future_downside_variance', 'downside_63']].mean().groupby('quintile').mean()
fig, ax = plt.subplots(figsize=(6.8, 3.1), layout='constrained')
for col, label, color in [('future_downside_variance','Realized next-window risk',COLORS[1]),('predicted_downside_variance','CatBoost forecast',COLORS[4]),('downside_63','Historical downside risk',COLORS[0])]:
    ax.plot(group.index, group[col]*1e4, label=label, color=color, marker='o')
ax.set(xlabel='Predicted-risk quintile within month', ylabel='Daily downside semivariance × 10⁴', xticks=range(1,6))
ax.grid(axis='y', alpha=.2)
ax.legend(frameon=False)
save(fig, 'risk_calibration')
print('Saved three publication figures as PNG and PDF.')
