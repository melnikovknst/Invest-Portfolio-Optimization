"""Monthly asset panel for a forward, purged downside-risk forecasting task."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
from .fundamentals import RATIOS,sha256

PRICE_FEATURES=['momentum_21','momentum_63','momentum_252','reversal_5','volatility_21','volatility_63','volatility_252','downside_63','skew_63','drawdown_63','drawdown_252','beta_126','residual_volatility_126','correlation_126','relative_momentum_63','baseline_variance','regime_variance','stress_probability','market_volatility','market_drawdown','market_correlation']
FUNDAMENTAL_FEATURES=RATIOS+['revenue_growth','log_assets','quality_score','quality_coverage']
EXTRA_FEATURES=['log_dollar_volume_21','relative_volume_21_63','amihud_21','intraday_range_21','report_age_days','filing_lag_days','annual_age_days','amended_fraction','known_filings','fundamental_missing_fraction','month_sin','month_cos']
FEATURE_SETS={'price_regime':PRICE_FEATURES,'plus_fundamentals':PRICE_FEATURES+FUNDAMENTAL_FEATURES,'full':PRICE_FEATURES+FUNDAMENTAL_FEATURES+EXTRA_FEATURES}
HORIZON=21

def load_ohlcv(root,prices):
 candidates=[Path(os.environ['PORTFOLIO_OHLCV'])] if os.environ.get('PORTFOLIO_OHLCV') else []
 candidates += [root/'data/SP500_Historical_Data.csv',Path.home()/'.cache/kagglehub/datasets/jacksaleeby/s-and-p500-historical-data/versions/1/SP500_Historical_Data.csv']
 source=next((p for p in candidates if p.exists()),None)
 if source is None:raise FileNotFoundError('Set PORTFOLIO_OHLCV to the source OHLCV CSV used by price_EDA.ipynb, or place it in data/SP500_Historical_Data.csv.')
 keep=[]
 for chunk in pd.read_csv(source,chunksize=200000,parse_dates=['Date']):
  keep.append(chunk[chunk.Ticker.isin(prices.columns)&chunk.Date.isin(prices.index)])
 f=pd.concat(keep,ignore_index=True)
 if f.duplicated(['Date','Ticker']).any():raise ValueError('Duplicate OHLCV keys')
 p=f.pivot(index='Date',columns='Ticker',values='Adj Close').reindex(index=prices.index,columns=prices.columns)
 if not np.allclose(p,prices,rtol=0,atol=1e-10,equal_nan=False):raise ValueError('OHLCV price snapshot differs from the supplied split files')
 for col in ['High','Low','Close','Volume']:
  if not np.isfinite(f[col]).all():raise ValueError(f'Invalid {col}')
 if not ((f.High>=f.Low)&(f.Low>0)&(f.Close>0)&(f.Volume>=0)).all():raise ValueError('Invalid OHLCV ranges')
 panels={c:f.pivot(index='Date',columns='Ticker',values=c).reindex(index=prices.index,columns=prices.columns) for c in ['High','Low','Close','Volume']}
 return panels,{'file_name':source.name,'sha256':sha256(source),'matching_price_cells':int(p.size),'source':'https://www.kaggle.com/datasets/jacksaleeby/s-and-p500-historical-data','data_version':1}

def make_feature_panel(prices,returns,fundamentals,market,regime_features,ohlcv):
 if not isinstance(returns.index,pd.DatetimeIndex) or returns.index.hasnans or not returns.index.is_unique or not returns.index.is_monotonic_increasing:
  raise ValueError('Return calendar must be unique and chronological')
 for name,frame in [('prices',prices),*ohlcv.items()]:
  if not frame.index.equals(returns.index) or not frame.columns.equals(returns.columns):
   raise ValueError(f'{name} must align with the return calendar and asset columns')
 if not returns.columns.is_unique or not market:
  raise ValueError('Unique assets and nonempty market snapshots are required')
 if not regime_features.index.is_unique or not regime_features.index.is_monotonic_increasing:
  raise ValueError('Regime feature index must be unique and chronological')
 if fundamentals.duplicated(['date','ticker']).any():
  raise ValueError('Duplicate fundamental snapshots')
 if 'information_date' not in fundamentals or not (fundamentals.information_date<fundamentals.date).all():
  raise ValueError('Fundamental information_date must strictly precede formation')
 if 'latest_used_available' in fundamentals:
  used=fundamentals.latest_used_available.notna()
  if not (fundamentals.loc[used,'latest_used_available']<=fundamentals.loc[used,'information_date']).all():
   raise ValueError('Fundamental filing is unavailable at the information cutoff')
 rows=[];market_return=returns.mean(axis=1)
 for date,m in market.items():
  j=returns.index.get_loc(date);history=returns.iloc[:j];p=prices.iloc[:j];assets=m['assets']
  trailing=history.tail(63);downside=np.minimum(trailing,0).pow(2).mean().clip(lower=1e-10)
  future=returns.iloc[j:j+HORIZON]
  label_end=future.index[-1] if len(future)==HORIZON else pd.NaT
  target=np.minimum(future,0).pow(2).mean().clip(lower=1e-10) if len(future)==HORIZON else pd.Series(np.nan,index=assets)
  row=pd.DataFrame(index=assets)
  for window in [21,63,252]:
   row[f'momentum_{window}']=(1+history.tail(window)).prod()-1
   row[f'volatility_{window}']=history.tail(window).std(ddof=1)*np.sqrt(252)
  row['reversal_5']=-((1+history.tail(5)).prod()-1)
  row['downside_63']=downside
  row['skew_63']=trailing.skew()
  for w in [63,252]:row[f'drawdown_{w}']=p.iloc[-1]/p.tail(w).max()-1
  mr=market_return.iloc[:j].tail(126);rh=history.tail(126)
  beta=rh.apply(lambda x:x.cov(mr))/mr.var(ddof=1)
  row['beta_126']=beta;row['correlation_126']=rh.corrwith(mr)
  row['residual_volatility_126']=(rh-mr.to_numpy()[:,None]*beta.to_numpy()[None,:]).std(ddof=1)*np.sqrt(252)
  row['relative_momentum_63']=row.momentum_63-((1+mr.tail(63)).prod()-1)
  row['baseline_variance']=np.diag(m['baseline']);row['regime_variance']=np.diag(m['regime'])
  row['stress_probability']=m['diagnostics']['forecast_stress_probability']
  rf=regime_features.loc[regime_features.index<date].iloc[-1]
  row['market_volatility']=rf.realized_volatility;row['market_drawdown']=rf.drawdown;row['market_correlation']=rf.average_correlation
  vol=ohlcv['Volume'].iloc[:j];dollar=vol*ohlcv['Close'].iloc[:j]
  row['log_dollar_volume_21']=np.log1p(dollar.tail(21).mean())
  row['relative_volume_21_63']=vol.tail(21).mean()/vol.tail(63).mean().replace(0,np.nan)
  row['amihud_21']=(history.abs()/dollar.replace(0,np.nan)).tail(21).mean()*1e6
  row['intraday_range_21']=((ohlcv['High'].iloc[:j]-ohlcv['Low'].iloc[:j])/ohlcv['Close'].iloc[:j]).tail(21).mean()
  row['month_sin']=np.sin(2*np.pi*date.month/12);row['month_cos']=np.cos(2*np.pi*date.month/12)
  row['date']=date;row['price_information_end']=history.index[-1];row['label_end']=label_end
  row['future_downside_variance']=target;row['target_log_risk_ratio']=np.log(target/downside)
  row=row.reset_index(names='ticker');rows.append(row)
 panel=pd.concat(rows,ignore_index=True)
 panel=panel.merge(fundamentals,on=['date','ticker'],how='left',validate='one_to_one',indicator=True)
 if not panel['_merge'].eq('both').all():raise ValueError('Missing fundamental snapshot')
 panel=panel.drop(columns='_merge')
 assert len(panel)==sum(len(m['assets']) for m in market.values())
 assert (panel.price_information_end<panel.date).all()
 mask=panel.label_end.notna();assert (panel.loc[mask,'label_end']>=panel.loc[mask,'date']).all()
 cols=FEATURE_SETS['full'];panel[cols]=panel[cols].replace([np.inf,-np.inf],np.nan)
 assert not set(cols)&{'label_end','future_downside_variance','target_log_risk_ratio'}
 return panel.sort_values(['date','ticker']).reset_index(drop=True)

def forecast_risk(panel,evaluation_dates,depth=4,feature_set='full',iterations=300,model_dir=None):
 """Expanding-window annual refits with fully matured 21-day labels only.

 Validation and test use the same learning algorithm. Test refits may use earlier
 test observations after their label windows have closed; hyperparameters stay fixed.
 """
 from catboost import CatBoostRegressor
 if panel.duplicated(['date','ticker']).any():raise ValueError('Duplicate forecasting rows')
 panel=panel.sort_values(['date','ticker'],kind='stable').reset_index(drop=True)
 features=FEATURE_SETS[feature_set]+['ticker'];outputs=[];audits=[];importance=[]
 evaluation_dates=pd.DatetimeIndex(evaluation_dates)
 if evaluation_dates.empty or evaluation_dates.hasnans or not evaluation_dates.is_unique or not evaluation_dates.is_monotonic_increasing:
  raise ValueError('Forecast dates must be nonempty, unique and chronological')
 if not evaluation_dates.isin(panel.date).all():raise ValueError('Missing forecast formation rows')
 for year in sorted(set(evaluation_dates.year)):
  dates=evaluation_dates[evaluation_dates.year==year];cutoff=dates.min()
  train=panel[(panel.date<cutoff)&(panel.label_end<cutoff)&panel.target_log_risk_ratio.notna()].copy()
  test=panel[panel.date.isin(dates)].copy()
  if len(train)<1000:raise ValueError(f'Insufficient matured training rows at {cutoff}')
  assert train.label_end.max()<cutoff<=test.date.min()
  model=CatBoostRegressor(iterations=iterations,depth=depth,learning_rate=.04,l2_leaf_reg=10,loss_function='RMSE',random_seed=20260906,thread_count=2,has_time=True,allow_writing_files=False,verbose=False)
  model.fit(train[features],train.target_log_risk_ratio,cat_features=['ticker'])
  # The bounds are fixed before validation: exponential residual-risk ratios [exp(-3), exp(3)].
  prediction=np.clip(model.predict(test[features]),-3,3)
  out=test[['date','ticker','label_end','future_downside_variance','downside_63']].copy()
  out['predicted_log_risk_ratio']=prediction
  out['predicted_downside_variance']=out.downside_63*np.exp(prediction)
  out['model_training_cutoff']=cutoff;outputs.append(out)
  audits.append({'year':year,'fit_date':cutoff,'training_rows':len(train),'training_dates':train.date.nunique(),'last_training_formation':train.date.max(),'last_training_label_end':train.label_end.max(),'prediction_dates':len(dates),'depth':depth,'iterations':iterations,'feature_set':feature_set,'features':len(features)})
  importance.append(pd.DataFrame({'feature':features,'importance':model.feature_importances_,'year':year,'feature_set':feature_set}))
  if model_dir:
   Path(model_dir).mkdir(parents=True,exist_ok=True);model.save_model(str(Path(model_dir)/f'{feature_set}_d{depth}_{year}.cbm'))
  print(f'CatBoost {feature_set}, depth {depth}: {year}; {len(train):,} matured training rows',flush=True)
 return pd.concat(outputs,ignore_index=True),pd.DataFrame(audits),pd.concat(importance,ignore_index=True)

def prediction_diagnostics(predictions):
 p=predictions.dropna(subset=['future_downside_variance']).copy();rows=[]
 for name,col in [('Historical downside','downside_63'),('CatBoost','predicted_downside_variance')]:
  dates=p.groupby('date').apply(lambda g:g[col].corr(g.future_downside_variance,method='spearman'),include_groups=False)
  rows.append({'model':name,'rows':len(p),'months':len(dates),'log_risk_RMSE':np.sqrt(np.mean((np.log(p[col])-np.log(p.future_downside_variance))**2)),'mean_monthly_rank_IC':dates.mean(),'median_monthly_rank_IC':dates.median()})
 return pd.DataFrame(rows).set_index('model')
