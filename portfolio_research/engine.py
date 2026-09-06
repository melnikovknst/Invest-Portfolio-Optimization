"""One return convention, one drift-aware execution engine, one metric schema."""
from __future__ import annotations
from dataclasses import asdict
import hashlib,json,pickle,platform
from pathlib import Path
import numpy as np
import pandas as pd
from .regimes import (BacktestConfig,RegimeConfig,BacktestResult,get_eligible_universe,
 build_regime_features,estimate_ledoit_wolf,estimate_regime_covariance,solve_gmv,make_rebalance_dates)
from .fundamentals import sha256

TRADING_DAYS=252
ROOT=Path(__file__).resolve().parents[1]
LABELS={'equal_weight':'Equal weight','baseline':'Baseline GMV','pipeline1':'Pipeline 1','pipeline2':'Pipeline 2','pipeline3':'Pipeline 3'}
METRIC_COLUMNS=['CAGR','Annualized Volatility','Sharpe','Sortino','Maximum Drawdown','CVaR 95%','CVaR 99%','Annualized Turnover','Average Effective Assets']

def load_prices(root=ROOT):
 frames=[];splits={};records=[];hashes={}
 for name,file in [('train','train.csv'),('validation','val.csv'),('test','test.csv')]:
  p=root/'data'/file;f=pd.read_csv(p,parse_dates=['Date'])
  if f.duplicated(['Date','Ticker']).any():raise ValueError(f'Duplicate date/ticker keys: {file}')
  if not np.isfinite(f['Adj Close']).all() or not (f['Adj Close']>0).all():raise ValueError('Invalid prices')
  splits[name]=pd.DatetimeIndex(sorted(f.Date.unique()))
  records.append({'split':name,'start':f.Date.min(),'end':f.Date.max(),'days':f.Date.nunique(),'assets':f.Ticker.nunique(),'rows':len(f)})
  frames.append(f);hashes[file]=sha256(p)
 assert splits['train'].max()<splits['validation'].min()<splits['validation'].max()<splits['test'].min()
 allp=pd.concat(frames)
 assert not allp.duplicated(['Date','Ticker']).any()
 prices=allp.pivot(index='Date',columns='Ticker',values='Adj Close').sort_index()
 if prices.isna().any().any():raise ValueError('The shared protocol requires the supplied complete panel.')
 returns=prices.pct_change(fill_method=None)
 return prices,returns,splits,pd.DataFrame(records).set_index('split'),hashes

def file_fingerprint(paths,settings):
 payload={'files':{str(Path(p).relative_to(ROOT)) if Path(p).is_relative_to(ROOT) else Path(p).name:sha256(p) for p in paths},'settings':settings}
 return hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest()

def prepare_market(prices,returns,dates,config=BacktestConfig(),regime=RegimeConfig(),cache_dir=None):
 """Cache formation-date covariances, not decisions or selected test results."""
 cache_dir=Path(cache_dir or ROOT/'artifacts/cache/market');cache_dir.mkdir(parents=True,exist_ok=True)
 data_hash=hashlib.sha256(pd.util.hash_pandas_object(returns,index=True).values.tobytes()).hexdigest()
 key=file_fingerprint([Path(__file__),ROOT/'portfolio_research/regimes.py'],{'data':data_hash,'config':asdict(config),'regime':asdict(regime)})
 features=build_regime_features(returns,regime);market={}
 for i,date in enumerate(dates):
  path=cache_dir/f'{key[:16]}_{date:%Y%m%d}.pkl'
  if path.exists():
   with path.open('rb') as f: market[date]=pickle.load(f)
  else:
   assets,history=get_eligible_universe(returns,prices,date,config)
   base,bd=estimate_ledoit_wolf(history)
   # An HMM failure stops the research run; there is no silent change of experiment.
   cov,diag=estimate_regime_covariance(history,features,date,regime)
   row={'assets':assets,'baseline':base,'regime':cov,'diagnostics':{**bd,**diag,'history_start':history.index.min(),'history_end':history.index.max(),'history_observations':len(history)}}
   market[date]=row
   with path.open('wb') as f:pickle.dump(row,f)
  if (i+1)%12==0 or i+1==len(dates):print(f'Market snapshots: {i+1}/{len(dates)}; latest {date.date()}',flush=True)
 return market,features,key

def run_backtest(returns,dates,market,config=BacktestConfig(),strategy='baseline',fundamentals=None,quality_strength=0.,predictions=None,ml_strength=0.):
 """Self-financing holdings drift daily; trades occur only on scheduled formation dates.

 Costs reduce wealth at rebalance before that day's return. L1 turnover includes both
 purchases and sales (initial cash investment has turnover one). Signals use prior
 closes; same-close execution is an idealization recorded in the protocol.
 """
 dates=pd.DatetimeIndex(dates);reb=set(make_rebalance_dates(dates,config.rebalance))
 w=pd.Series(dtype=float); daily=[];targets=[];diagnostics=[]
 for date in dates:
  turnover=cost=0.
  if date in reb:
   m=market[date];assets=m['assets'];cov=m['baseline'].copy() if strategy in ('baseline','equal_weight') else m['regime'].copy()
   extra={}; q=pd.Series(.5,index=assets)
   if fundamentals is not None:
    q=fundamentals.loc[fundamentals.date.eq(date)].set_index('ticker').quality_score.reindex(assets)
    if q.isna().any():raise ValueError('Missing point-in-time quality snapshot')
   if strategy in ('pipeline2','pipeline3'):
    if fundamentals is None:raise ValueError('Quality strategies require fundamentals')
    penalty=quality_strength*float(np.median(np.diag(m['regime'])))*(1-q.to_numpy())
    cov+=np.diag(penalty)
   if strategy=='pipeline3':
    p=predictions.loc[predictions.date.eq(date)].set_index('ticker').predicted_downside_variance.reindex(assets)
    if p.isna().any() or not np.isfinite(p).all() or (p<0).any():raise ValueError('Invalid ML risk predictions')
    cov+=ml_strength*np.diag(p.to_numpy());extra['mean_predicted_downside_variance']=p.mean()
   if strategy=='equal_weight':
    target=pd.Series(1/len(assets),index=assets);solver={'solver_success':True,'fallback_used':False}
   else:target,solver=solve_gmv(cov,assets,config.max_weight)
   if not solver['solver_success']:raise RuntimeError(f'Optimization failed at {date}: {solver}')
   union=w.index.union(target.index);turnover=float((target.reindex(union,fill_value=0)-w.reindex(union,fill_value=0)).abs().sum())
   cost=config.cost_rate*turnover
   if cost>=1:raise ValueError('Trading costs exhaust capital')
   w=target.copy();full=target.reindex(returns.columns,fill_value=0);full.name=date;targets.append(full)
   d={'date':date,**m['diagnostics'],**solver,**extra,'turnover':turnover,'transaction_cost':cost,'eligible_assets':len(assets),'active_positions':int((target>1e-8).sum()),'maximum_weight':target.max(),'hhi':float(target.pow(2).sum()),'effective_assets':float(1/target.pow(2).sum()),'portfolio_quality':float(target.dot(q))}
   diagnostics.append(d)
  r=returns.loc[date,w.index]
  if r.isna().any():raise ValueError(f'Missing held-security returns on {date}')
  gross=float(w.dot(r));net=(1-cost)*(1+gross)-1
  if net<=-1:raise ValueError('Portfolio wealth is non-positive')
  daily.append({'date':date,'gross_return':gross,'net_return':net,'turnover':turnover,'transaction_cost':cost})
  # Essential correction: tomorrow's returns must use today's drifted holdings.
  w=w*(1+r)/(1+gross)
  assert abs(w.sum()-1)<1e-9
 result=BacktestResult(strategy,config,pd.DataFrame(daily).set_index('date'),pd.DataFrame(targets).rename_axis('rebalance_date'),pd.DataFrame(diagnostics).set_index('date'))
 validate_result(result)
 return result

def drawdown(returns):
 wealth=(1+returns).cumprod();peak=wealth.cummax().clip(lower=1.)
 return wealth/peak-1

def var_cvar(returns,alpha):
 loss=-np.asarray(returns);cutoff=np.quantile(loss,alpha)
 return float(cutoff),float(loss[loss>=cutoff].mean())

def metrics(result,return_column='net_return'):
 r=result.returns[return_column];years=len(r)/252;wealth=(1+r).prod();vol=r.std(ddof=1)*np.sqrt(252)
 dd=drawdown(r);down=np.sqrt(np.mean(np.minimum(r,0)**2))*np.sqrt(252);mean=r.mean()*252
 var95,cvar95=var_cvar(r,.95);var99,cvar99=var_cvar(r,.99);d=result.diagnostics
 underwater=(dd<0).astype(int);runs=underwater.groupby((underwater==0).cumsum()).sum()
 out={'Cumulative Return':wealth-1,'CAGR':wealth**(1/years)-1,'Average Daily Return':r.mean(),'Arithmetic Annual Return':mean,'Annualized Volatility':vol,'Downside Deviation':down,'Sharpe':mean/vol if vol>0 else np.nan,'Sortino':mean/down if down>0 else np.nan,'Maximum Drawdown':-dd.min(),'Drawdown Duration':runs.max(),'VaR 95%':var95,'CVaR 95%':cvar95,'VaR 99%':var99,'CVaR 99%':cvar99,'Annualized Turnover':result.returns.turnover.sum()/years,'Average Turnover':d.turnover.mean(),'Total Transaction Costs':result.returns.transaction_cost.sum(),'Rebalances':len(d),'Average HHI':d.hhi.mean(),'Average Effective Assets':d.effective_assets.mean(),'Average Active Positions':d.active_positions.mean(),'Average Maximum Weight':d.maximum_weight.mean(),'Mean Target L1 Change':result.target_weights.diff().abs().sum(axis=1).iloc[1:].mean(),'Solver Failures':int((~d.solver_success).sum()),'HMM Fallbacks':int(d.get('hmm_fallback',pd.Series(False,index=d.index)).sum()),'Average Portfolio Quality':d.portfolio_quality.mean()}
 out['Calmar']=out['CAGR']/out['Maximum Drawdown'] if out['Maximum Drawdown']>0 else np.nan
 return pd.Series(out)

def metric_table(results):return pd.DataFrame({k:metrics(v) for k,v in results.items()}).T

def risk_score(table):
 weights={'Annualized Volatility':.30,'CVaR 95%':.25,'Maximum Drawdown':.20,'Annualized Turnover':.10,'Average HHI':.10,'Sharpe':.05}
 return sum(w*table[k].rank(ascending=k!='Sharpe',method='average') for k,w in weights.items())

def validate_result(result):
 d=result.diagnostics;w=result.target_weights
 assert (d.history_end<d.index).all()
 if 'hmm_history_end' in d:assert (d.hmm_history_end<d.index).all()
 assert np.allclose(w.sum(axis=1),1,atol=1e-7)
 assert w.min().min()>=-1e-8 and w.max().max()<=result.config.max_weight+1e-7
 assert np.isfinite(result.returns.to_numpy()).all()
 assert np.allclose(result.returns.net_return,(1-result.returns.transaction_cost)*(1+result.returns.gross_return)-1)
 assert np.allclose(result.returns.transaction_cost,result.returns.turnover*result.config.cost_rate)

def export_result(result,name,split,root=ROOT):
 folder=root/'artifacts/results'/split/name;folder.mkdir(parents=True,exist_ok=True)
 result.returns.to_csv(folder/'daily_returns.csv');result.target_weights.to_csv(folder/'target_weights.csv');result.diagnostics.to_csv(folder/'diagnostics.csv')
 metrics(result).rename('value').to_csv(folder/'metrics.csv')
 (folder/'config.json').write_text(json.dumps(asdict(result.config),indent=2))

def load_result(name,split,root=ROOT):
 f=root/'artifacts/results'/split/name
 r=pd.read_csv(f/'daily_returns.csv',index_col=0,parse_dates=True);w=pd.read_csv(f/'target_weights.csv',index_col=0,parse_dates=True);d=pd.read_csv(f/'diagnostics.csv',index_col=0,parse_dates=True)
 for c in d.columns:
  if c.endswith('_start') or c.endswith('_end'):d[c]=pd.to_datetime(d[c])
 return BacktestResult(name,BacktestConfig(**json.loads((f/'config.json').read_text())),r,w,d)

def core_metrics(x):
 x=np.asarray(x);wealth=np.cumprod(1+x);peak=np.maximum.accumulate(np.r_[1.,wealth])[1:]
 return np.array([x.std(ddof=1)*np.sqrt(252),var_cvar(x,.95)[1],-np.min(wealth/peak-1),x.mean()/x.std(ddof=1)*np.sqrt(252)])

def paired_bootstrap(base,experiment,replications=2000,block=20,seed=20260906):
 assert base.index.equals(experiment.index)
 values=np.column_stack([base,experiment]);n=len(values);rng=np.random.default_rng(seed);out=[]
 for _ in range(replications):
  starts=rng.integers(0,n-block+1,size=int(np.ceil(n/block)))
  ix=(starts[:,None]+np.arange(block)).ravel()[:n]
  out.append(core_metrics(values[ix,1])-core_metrics(values[ix,0]))
 out=np.asarray(out)
 return pd.DataFrame({'difference':core_metrics(values[:,1])-core_metrics(values[:,0]),'lower_95':np.quantile(out,.025,axis=0),'upper_95':np.quantile(out,.975,axis=0)},index=['Annualized Volatility','CVaR 95%','Maximum Drawdown','Sharpe'])
