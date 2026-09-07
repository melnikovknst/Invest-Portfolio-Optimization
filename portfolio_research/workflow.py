"""Reproducible experiment stages; validation freezes all choices before test allocation."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json,pickle,sys,hashlib
import numpy as np
import pandas as pd
from .engine import *
from .fundamentals import read_sec_facts,build_fundamental_panel,LINEAGE
from .features import load_ohlcv,make_feature_panel,forecast_risk,FEATURE_SETS

CONFIG=BacktestConfig()  # Original frozen LB252 / monthly / 15% cap / 10 bps.
QUALITY_GRID=[0.,.25,.5,1.,2.]
ML_GRID=[0.,.25,.5,1.]
DEPTH_GRID=[4,6]

def context():
 prices,returns,splits,summary,hashes=load_prices()
 return {'prices':prices,'returns':returns,'splits':splits,'summary':summary,'hashes':hashes,'config':CONFIG}

def write_json(path,value):
 Path(path).parent.mkdir(parents=True,exist_ok=True)
 Path(path).write_text(json.dumps(value,indent=2,default=str,sort_keys=True)+'\n')

def prepare_context():
 c=context();root=ROOT;cache=root/'artifacts/cache';cache.mkdir(parents=True,exist_ok=True)
 dates=make_rebalance_dates(c['prices'].index[c['prices'].index>='2010-01-01'],'M')
 market,features,key=prepare_market(c['prices'],c['returns'],dates,CONFIG)
 c.update(market=market,regime_features=features,market_key=key,formation_dates=dates)
 return c

def prepare_fundamentals(c):
 root=ROOT;cache=root/'artifacts/cache'
 facts,sources,audit=read_sec_facts(root/'data',root/'reference/ticker_cik.json')
 key=file_fingerprint([root/'portfolio_research/fundamentals.py',root/'reference/ticker_cik.json'],{'sources':sources.sha256.tolist(),'dates':c['formation_dates'].astype(str).tolist(),'trading_dates':c['prices'].index.astype(str).tolist()})
 manifest=cache/'fundamentals_cache.json';p=cache/'fundamental_panel.pkl'
 if p.exists() and manifest.exists() and json.loads(manifest.read_text()).get('key')==key:
  fundamentals=pd.read_pickle(p)
 else:
  fundamentals=build_fundamental_panel(facts,c['formation_dates'],c['prices'].columns,c['prices'].index)
  fundamentals.to_pickle(p);write_json(manifest,{'key':key})
 tables=root/'artifacts/tables';tables.mkdir(parents=True,exist_ok=True)
 sources.to_csv(tables/'fundamental_sources.csv',index=False)
 pd.DataFrame([audit]).to_csv(tables/'fundamental_fact_audit.csv',index=False)
 fundamentals.groupby(fundamentals.date.dt.year)[['quality_coverage','quality_score']].mean().to_csv(tables/'fundamental_coverage_by_year.csv')
 write_json(root/'reference/issuer_lineage.json',LINEAGE)
 c.update(fundamentals=fundamentals,fundamental_key=key,fact_audit=audit,sources=sources)
 return c

def prepare_features(c):
 ohlcv,source=load_ohlcv(ROOT,c['prices'])
 panel=make_feature_panel(c['prices'],c['returns'],c['fundamentals'],c['market'],c['regime_features'],ohlcv)
 panel.to_pickle(ROOT/'artifacts/cache/ml_panel.pkl');write_json(ROOT/'artifacts/tables/ohlcv_provenance.json',source)
 c.update(panel=panel,ohlcv_source=source)
 return c

def strategy(c,split,name,quality_strength=0,predictions=None,ml_strength=0,config=CONFIG):
 return run_backtest(c['returns'],c['splits'][split],c['market'],config,name,c.get('fundamentals'),quality_strength,predictions,ml_strength)

def run_controls(c,split):
 results={name:strategy(c,split,name) for name in ['equal_weight','baseline','pipeline1']}
 for name,r in results.items():export_result(r,name,split)
 return results

def verify_saved_result(result,name,split,root=ROOT):
 """Check notebook calculations without replacing the driver's saved evidence.

 Price-only notebooks have no SEC panel, so their neutral quality placeholder
 must not overwrite the driver's target-weighted quality exposure.
 """
 recorded=load_result(name,split,root=root)
 if asdict(result.config)!=asdict(recorded.config):
  raise ValueError(f'{split}/{name}: allocation settings differ from saved evidence')
 for field in ('returns','target_weights'):
  actual,expected=getattr(result,field),getattr(recorded,field)
  if not actual.index.equals(expected.index) or not actual.columns.equals(expected.columns):
   raise ValueError(f'{split}/{name}: {field} labels differ from saved evidence')
  if not np.allclose(actual.to_numpy(),expected.to_numpy(),rtol=0,atol=1e-9,equal_nan=True):
   raise ValueError(f'{split}/{name}: {field} differ from saved evidence')
 columns=metrics(recorded).index.drop('Average Portfolio Quality')
 if not np.allclose(metrics(result)[columns],metrics(recorded)[columns],rtol=0,atol=1e-9,equal_nan=True):
  raise ValueError(f'{split}/{name}: metrics differ from saved evidence')
 return recorded

def select_quality(c):
 results={f'lambda={v:g}':strategy(c,'validation','pipeline2',quality_strength=v) for v in QUALITY_GRID}
 table=metric_table(results);table['quality_strength']=QUALITY_GRID;table['risk_score']=risk_score(table)
 # Tie-break toward less intervention, including the explicit zero-strength control.
 chosen=table.sort_values(['risk_score','quality_strength'],kind='stable').iloc[0]
 table.to_csv(ROOT/'artifacts/tables/pipeline2_validation_grid.csv')
 selected=float(chosen.quality_strength)
 export_result(results[f'lambda={selected:g}'],'pipeline2','validation')
 return selected,table,results

def select_ml(c,quality_strength):
 dates=make_rebalance_dates(c['splits']['validation'],'M');rows=[];runs={};forecasts={}
 for depth in DEPTH_GRID:
  pred,audit,imp=forecast_risk(c['panel'],dates,depth=depth,feature_set='full')
  forecasts[depth]=(pred,audit,imp)
  for strength in ML_GRID:
   if strength==0 and depth!=DEPTH_GRID[0]:continue  # The no-ML control is counted once.
   key=f'depth={depth};eta={strength:g}'
   result=strategy(c,'validation','pipeline3',quality_strength,pred,strength);runs[key]=result
   rows.append({'candidate':key,'depth':depth,'ml_strength':strength,**metrics(result).to_dict()})
 table=pd.DataFrame(rows).set_index('candidate');table['risk_score']=risk_score(table)
 selected=table.sort_values(['risk_score','ml_strength','depth'],kind='stable').iloc[0]
 depth=int(selected.depth);eta=float(selected.ml_strength)
 table.to_csv(ROOT/'artifacts/tables/pipeline3_validation_grid.csv')
 pred,audit,imp=forecasts[depth]
 pred.to_csv(ROOT/'artifacts/tables/pipeline3_validation_predictions.csv',index=False)
 audit.to_csv(ROOT/'artifacts/tables/pipeline3_validation_training_audit.csv',index=False)
 imp.to_csv(ROOT/'artifacts/tables/pipeline3_validation_importance.csv',index=False)
 export_result(runs[f'depth={depth};eta={eta:g}'],'pipeline3','validation')
 return {'depth':depth,'ml_strength':eta,'iterations':300,'feature_set':'full'},table,runs,forecasts

def code_hashes():
 return {p.name:sha256(p) for p in sorted((ROOT/'portfolio_research'/name for name in ['engine.py','regimes.py','fundamentals.py','features.py','workflow.py']))}

def freeze(c,quality_strength,ml_spec):
 spec={'schema_version':1,'baseline':asdict(CONFIG),'regime':asdict(RegimeConfig()),'quality_strength':quality_strength,'ml':ml_spec,'quality_candidates':QUALITY_GRID,'ml_candidates':ML_GRID,'depth_candidates':DEPTH_GRID,'features':FEATURE_SETS,'input_hashes':c['hashes'],'ohlcv_source':c['ohlcv_source'],'fundamental_key':c['fundamental_key'],'market_key':c['market_key'],'code_hashes':code_hashes(),'selection_split':'2017-2020','test_range':[str(c['splits']['test'].min().date()),str(c['splits']['test'].max().date())],'cost_convention':'multiplicative upfront proportional costs; two-sided L1 trades versus fully drifted holdings','forecast_protocol':'expanding annual refits, label_end strictly before fit date; no test-dependent hyperparameter selection','historical_test_exposure':'The original baseline notebook already displayed this test. It is held out for new hyperparameter selection, not a pristine preregistered replication.'}
 write_json(ROOT/'artifacts/frozen_spec.json',spec)
 return spec

def verify_frozen(c,spec,require_features=False):
 """Reject changed inputs/settings even when Python assertions are disabled.

 The quality-only notebook deliberately has no OHLCV/ML panel. Full test
 execution requires those inputs; its provenance must never be optional.
 """
 checks=[
  ('input_hashes',c['hashes'],'Price inputs changed after freeze'),
  ('code_hashes',code_hashes(),'Shared implementation changed after freeze; repeat validation'),
  ('selection_split','2017-2020','Unexpected validation selection split'),
  ('baseline',asdict(CONFIG),'Baseline settings changed after freeze'),
  ('regime',asdict(RegimeConfig()),'Regime settings changed after freeze'),
  ('features',FEATURE_SETS,'Feature definitions changed after freeze'),
  ('fundamental_key',c['fundamental_key'],'Fundamental inputs changed after freeze'),
  ('market_key',c['market_key'],'Market snapshots changed after freeze'),
  ('test_range',[str(c['splits']['test'].min().date()),str(c['splits']['test'].max().date())],
   'Test range changed after freeze'),
 ]
 for key,expected,message in checks:
  if spec.get(key)!=expected:raise ValueError(message)
 if 'config' in c and asdict(c['config'])!=spec['baseline']:
  raise ValueError('Context allocation settings differ from the frozen baseline')
 if require_features and ('ohlcv_source' not in c or 'panel' not in c):
  raise ValueError('Full test execution requires the feature panel and OHLCV provenance')
 if 'ohlcv_source' in c and spec.get('ohlcv_source')!=c['ohlcv_source']:
  raise ValueError('OHLCV inputs changed after freeze')
 ml=spec.get('ml',{})
 if ml.get('feature_set') not in FEATURE_SETS:
  raise ValueError('Unknown frozen ML feature set')

def run_test(c,spec):
 verify_frozen(c,spec,require_features=True);results=run_controls(c,'test')
 q=spec['quality_strength'];ml=spec['ml']
 results['pipeline2']=strategy(c,'test','pipeline2',q)
 dates=make_rebalance_dates(c['splits']['test'],'M')
 pred,audit,imp=forecast_risk(c['panel'],dates,depth=ml['depth'],feature_set=ml['feature_set'],iterations=ml['iterations'],model_dir=ROOT/'artifacts/models')
 results['pipeline3']=strategy(c,'test','pipeline3',q,pred,ml['ml_strength'])
 for name,r in results.items():export_result(r,name,'test')
 metric_table(results).to_csv(ROOT/'artifacts/tables/test_metrics.csv')
 for name,f in [('predictions',pred),('training_audit',audit),('importance',imp)]:f.to_csv(ROOT/f'artifacts/tables/pipeline3_test_{name}.csv',index=False)
 return results,pred,audit,imp
