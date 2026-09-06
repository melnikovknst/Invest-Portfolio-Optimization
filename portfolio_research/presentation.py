"""Notebook-native, exportable figures and compact tables; no hidden calculations."""
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import PercentFormatter
from .engine import ROOT,LABELS,metrics,drawdown,BacktestResult
COLORS={'equal_weight':'#71717a','baseline':'#2563a6','pipeline1':'#c08a24','pipeline2':'#778d39','pipeline3':'#b75a86'}
LINES={'equal_weight':':','baseline':'-','pipeline1':'--','pipeline2':'-.','pipeline3':'-'}
PERCENT=['Cumulative Return','CAGR','Average Daily Return','Arithmetic Annual Return','Annualized Volatility','Downside Deviation','Maximum Drawdown','VaR 95%','CVaR 95%','VaR 99%','CVaR 99%','Total Transaction Costs','Average Maximum Weight','Average HHI']

def setup():
 plt.style.use('seaborn-v0_8-whitegrid')
 plt.rcParams.update({'figure.dpi':110,'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'grid.alpha':.25,'figure.figsize':(11,5),'savefig.facecolor':'white'})
 pd.set_option('display.max_columns',35);pd.set_option('display.max_rows',25)

def styled(table):
 return table.style.format({c:'{:.2%}' if c in PERCENT else '{:.3f}' for c in table.select_dtypes(include='number').columns},na_rep='N/A')

def save(fig,name):
 p=ROOT/'artifacts/figures';p.mkdir(parents=True,exist_ok=True)
 fig.savefig(p/f'{name}.png',dpi=170,bbox_inches='tight')
 fig.savefig(p/f'{name}.pdf',bbox_inches='tight')
 plt.show()

def performance_plot(results,name):
 fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,layout='constrained',height_ratios=[2,1])
 for key,result in results.items():
  r=result.returns.net_return;color=COLORS.get(key,'#2563a6');label=LABELS.get(key,key)
  axes[0].plot(r.index,(1+r).cumprod(),label=label,color=color,ls=LINES.get(key,'-'),lw=1.7)
  axes[1].plot(r.index,drawdown(r),color=color,ls=LINES.get(key,'-'),lw=1.2)
 first=next(iter(results.values())).returns.index
 axes[0].set(title=f'Net portfolio wealth | {first.min():%Y-%m-%d} to {first.max():%Y-%m-%d}',ylabel='Growth of 1 unit of initial wealth')
 axes[0].legend(ncol=min(5,len(results)),loc='upper left');axes[1].set(ylabel='Drawdown',xlabel='Date');axes[1].yaxis.set_major_formatter(PercentFormatter(1))
 save(fig,name)

def risk_plot(table,name):
 cols=['Annualized Volatility','CVaR 95%','Maximum Drawdown'];fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
 labels=[LABELS.get(x,x).replace(' ','\n',1) for x in table.index]
 for ax,col in zip(axes,cols):
  bars=ax.bar(labels,table[col],color=[COLORS.get(x,'#2563a6') for x in table.index],width=.65)
  ax.bar_label(bars,labels=[f'{v:.2%}' for v in table[col]],fontsize=9,padding=4)
  ax.set_title(col);ax.set_ylim(0,table[col].max()*1.22);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='x',visible=False)
 save(fig,name)

def quality_plot(panel,name):
 pivot=panel.pivot(index='ticker',columns='date',values='quality_score');coverage=panel.groupby(panel.date.dt.year).quality_coverage.mean()
 fig,axes=plt.subplots(1,2,figsize=(13,9),layout='constrained',width_ratios=[2.5,1])
 im=axes[0].imshow(pivot,aspect='auto',cmap='Blues',vmin=0,vmax=1,interpolation='nearest')
 axes[0].set_yticks(range(len(pivot)),pivot.index,fontsize=7)
 ticks=np.arange(0,pivot.shape[1],24);axes[0].set_xticks(ticks,[d.strftime('%Y') for d in pivot.columns[ticks]])
 axes[0].set(title='Point-in-time corporate quality',xlabel='Portfolio formation year',ylabel='Ticker');fig.colorbar(im,ax=axes[0],fraction=.03,label='Quality score (0–1)')
 axes[1].barh(coverage.index.astype(str),coverage,color='#2563a6');axes[1].set(xlim=(0,1),xlabel='Fraction of available ratios',title='Feature coverage by year');axes[1].xaxis.set_major_formatter(PercentFormatter(1))
 save(fig,name)

def allocation_plot(result,name):
 w=result.target_weights;assets=w.mean().nlargest(20).index
 fig,axes=plt.subplots(1,2,figsize=(13,7),layout='constrained',width_ratios=[1,2])
 w.mean().loc[assets].sort_values().plot.barh(ax=axes[0],color=COLORS.get(result.strategy,'#2563a6'))
 axes[0].set(title='20 largest average allocations',xlabel='Mean target weight');axes[0].xaxis.set_major_formatter(PercentFormatter(1))
 image=axes[1].imshow(w[assets].T,aspect='auto',cmap='Blues',vmin=0,vmax=result.config.max_weight,interpolation='nearest')
 axes[1].set_yticks(range(len(assets)),assets);ticks=np.arange(0,len(w),12);axes[1].set_xticks(ticks,[d.strftime('%Y') for d in w.index[ticks]])
 axes[1].set(title='Target weights at monthly rebalances',xlabel='Formation year');fig.colorbar(image,ax=axes[1],fraction=.025,label='Weight')
 save(fig,name)

def importance_plot(importance,name):
 x=importance.groupby('feature').importance.mean().nlargest(16).sort_values()
 fig,ax=plt.subplots(figsize=(10,6),layout='constrained');x.plot.barh(ax=ax,color='#2563a6');ax.set(title='CatBoost feature importance across annual fits',xlabel='Mean prediction-value-change importance',ylabel='')
 save(fig,name)

def annual_metrics(results):
 rows=[]
 for name,result in results.items():
  for year,g in result.returns.groupby(result.returns.index.year):
   r=g.net_return
   rows.append({'strategy':name,'year':year,'days':len(r),'return':(1+r).prod()-1,'volatility':r.std(ddof=1)*np.sqrt(252),'daily_CVaR95':-r[r<=r.quantile(.05)].mean()})
 return pd.DataFrame(rows)

def cost_sensitivity(results,bps_values=(0,10,25,50)):
 rows=[]
 for name,result in results.items():
  for bps in bps_values:
   r=copy.deepcopy(result);r.returns['transaction_cost']=r.returns.turnover*bps/10000
   r.returns['net_return']=(1-r.returns.transaction_cost)*(1+r.returns.gross_return)-1
   rows.append({'strategy':name,'cost_bps':bps,**metrics(r).to_dict()})
 return pd.DataFrame(rows)

def regime_conditioned(results):
 rows=[]
 for name,result in results.items():
  p=result.diagnostics.forecast_stress_probability.reindex(result.returns.index).ffill()
  for label,mask in [('Calm forecast',p<.5),('Stress forecast',p>=.5)]:
   r=result.returns.net_return[mask]
   rows.append({'strategy':name,'regime':label,'days':len(r),'Annualized Volatility':r.std(ddof=1)*np.sqrt(252),'Sharpe':r.mean()/r.std(ddof=1)*np.sqrt(252),'CVaR 95%':-r[r<=r.quantile(.05)].mean()})
 return pd.DataFrame(rows)
