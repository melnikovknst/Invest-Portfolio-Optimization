"""Build features, select only on validation, freeze, evaluate, and persist all experiments."""
from pathlib import Path
import os,sys,json,pickle,copy
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/portfolio-mpl')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from portfolio_research.workflow import *
from portfolio_research.features import prediction_diagnostics
from portfolio_research.presentation import cost_sensitivity,annual_metrics,regime_conditioned

c=prepare_features(prepare_fundamentals(prepare_context()))
with (ROOT/'artifacts/cache/context.pkl').open('wb') as f:pickle.dump(c,f)
controls=run_controls(c,'validation');q,qgrid,qresults=select_quality(c)
ml,mlgrid,mlresults,forecasts=select_ml(c,q)
print('Selected on validation:',q,ml,flush=True)
# Ablation model forms and strengths are fixed here, before any test evaluation.
ablations={}
for fs in ['price_regime','plus_fundamentals']:
 pred,audit,imp=forecast_risk(c['panel'],make_rebalance_dates(c['splits']['validation'],'M'),depth=ml['depth'],feature_set=fs)
 r=strategy(c,'validation','pipeline3',q,pred,ml['ml_strength']);export_result(r,'pipeline3_'+fs,'validation')
 ablations[fs]=metrics(r)
 pd.DataFrame(audit).to_csv(ROOT/f'artifacts/tables/ablation_{fs}_validation_training.csv',index=False)
pd.DataFrame(ablations).T.to_csv(ROOT/'artifacts/tables/pipeline3_validation_ablations.csv')
# Monthly publication-lag sensitivity is defined on validation, with the primary strength fixed.
lagged=c['fundamentals'].copy();lagged['quality_score']=lagged.groupby('ticker').quality_score.shift(1).fillna(.5)
lag_context={**c,'fundamentals':lagged}
lag_result=strategy(lag_context,'validation','pipeline2',q);export_result(lag_result,'pipeline2_extra_month_lag','validation')
spec=freeze(c,q,ml)
print('FROZEN SPECIFICATION SAVED. Beginning held-out evaluation.',flush=True)
results,pred,audit,imp=run_test(c,spec)
print(metric_table(results)[METRIC_COLUMNS].to_string(),flush=True)
for fs in ['price_regime','plus_fundamentals']:
 ap,aa,ai=forecast_risk(c['panel'],make_rebalance_dates(c['splits']['test'],'M'),depth=ml['depth'],feature_set=fs)
 ar=strategy(c,'test','pipeline3',q,ap,ml['ml_strength']);export_result(ar,'pipeline3_'+fs,'test')
 ap.to_csv(ROOT/f'artifacts/tables/ablation_{fs}_test_predictions.csv',index=False)
lag_result=strategy(lag_context,'test','pipeline2',q);export_result(lag_result,'pipeline2_extra_month_lag','test')
comparisons=[('baseline','pipeline1'),('pipeline1','pipeline2'),('pipeline2','pipeline3'),('baseline','pipeline2'),('baseline','pipeline3')]
ci=[]
for a,b in comparisons:
 for block in [10,20,60]:
  t=paired_bootstrap(results[a].returns.net_return,results[b].returns.net_return,replications=2000,block=block).reset_index(names='metric')
  t['reference']=a;t['experiment']=b;t['block_length']=block;ci.append(t)
pd.concat(ci,ignore_index=True).to_csv(ROOT/'artifacts/tables/test_paired_bootstrap.csv',index=False)
cost_sensitivity(results).to_csv(ROOT/'artifacts/tables/test_cost_sensitivity.csv',index=False)
annual_metrics(results).to_csv(ROOT/'artifacts/tables/test_annual_metrics.csv',index=False)
regime_conditioned(results).to_csv(ROOT/'artifacts/tables/test_regime_conditioned.csv',index=False)
prediction_diagnostics(pred).to_csv(ROOT/'artifacts/tables/test_prediction_metrics.csv')
val_results={name:load_result(name,'validation') for name in LABELS}
metric_table(val_results).to_csv(ROOT/'artifacts/tables/validation_metrics.csv')
# Independent zero-strength ablation checks must recover the preceding pipeline.
p10=strategy(c,'test','pipeline2',0)
p20=strategy(c,'test','pipeline3',q,pred,0)
assert np.allclose(p10.returns.net_return,results['pipeline1'].returns.net_return,atol=1e-11)
assert np.allclose(p20.returns.net_return,results['pipeline2'].returns.net_return,atol=1e-11)
write_json(ROOT/'artifacts/run_manifest.json',{'status':'experiments_completed','python':sys.version,'code_hashes':code_hashes(),'input_hashes':c['hashes'],'frozen_spec_sha256':sha256(ROOT/'artifacts/frozen_spec.json'),'test_days':len(c['splits']['test']),'test_tail_observations_95':int(np.ceil(len(c['splits']['test'])*.05)),'notebook_execution':'pending','zero_strength_controls_verified':True,'validation_selection':{'quality_strength':q,**ml}})
print('ALL EXPERIMENTS COMPLETED',flush=True)
