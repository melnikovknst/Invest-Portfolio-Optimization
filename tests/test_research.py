"""Financial invariants and point-in-time edge cases, rather than implementation snapshots."""
import unittest
import numpy as np
import pandas as pd
from portfolio_research.engine import run_backtest,BacktestConfig,drawdown,var_cvar
from portfolio_research.fundamentals import company_snapshot,quality_scores,RATIOS

class ResearchTests(unittest.TestCase):
 def test_buy_and_hold_drift_and_multiplicative_cost(self):
  dates=pd.date_range('2020-01-02',periods=3,freq='B')
  r=pd.DataFrame({'A':[.1,.1,0.],'B':[0.,0.,.2]},index=dates)
  market={dates[0]:{'assets':['A','B'],'baseline':np.eye(2),'regime':np.eye(2),'diagnostics':{'history_end':pd.Timestamp('2019-12-31')}}}
  cfg=BacktestConfig(max_weight=1,transaction_cost_bps=10)
  result=run_backtest(r,dates,market,cfg,'equal_weight')
  expected=.999*(.5*np.prod(1+r.A)+.5*np.prod(1+r.B))
  self.assertAlmostEqual(np.prod(1+result.returns.net_return),expected,places=12)
  self.assertAlmostEqual(result.returns.gross_return.iloc[1],.1*1.1/2.1,places=12)
  self.assertAlmostEqual(result.returns.turnover.sum(),1.)
 def test_drawdown_includes_initial_capital(self):
  self.assertAlmostEqual(-drawdown(pd.Series([-.1,0.,.01])).min(),.1)
 def test_cvar_is_daily_loss_tail(self):
  r=pd.Series([-.2,-.1,0.,.1]);v,c=var_cvar(r,.75)
  self.assertAlmostEqual(v,.125);self.assertAlmostEqual(c,.2)
 def test_filing_embargo_and_future_fact(self):
  facts=pd.DataFrame([{'ticker':'A','metric':'assets','value':100.,'end':pd.Timestamp('2019-12-31'),'filed':pd.Timestamp('2020-02-03'),'available':pd.Timestamp('2020-02-04'),'priority':0,'accn':'x','form':'10-K'}])
  a=company_snapshot(facts,'A','2020-02-04');b=company_snapshot(facts,'A','2020-02-05')
  self.assertTrue(np.isnan(a['log_assets']));self.assertAlmostEqual(b['log_assets'],np.log(100))
  future=facts.copy();future['available']=pd.Timestamp('2022-01-01');future['filed']=pd.Timestamp('2021-12-31');future['value']=1e12
  both=pd.concat([facts,future]);c=company_snapshot(both,'A','2020-02-05')
  self.assertEqual(b,c)
 def test_absent_fundamentals_are_neutral(self):
  s=pd.DataFrame({x:[np.nan,np.nan] for x in RATIOS});s['fundamental_missing_fraction']=1.
  q=quality_scores(s);self.assertTrue(np.allclose(q.quality_score,.5));self.assertTrue(np.allclose(q.quality_coverage,0))


class CalendarAndForecastTests(unittest.TestCase):
 def test_exchange_holiday_cannot_advance_publication(self):
  facts=pd.DataFrame([{'ticker':'A','metric':'assets','value':100.,'end':pd.Timestamp('2019-06-30'),'filed':pd.Timestamp('2019-08-30'),'available':pd.Timestamp('2019-09-02'),'priority':0,'accn':'x','form':'10-Q'}])
  a=company_snapshot(facts,'A','2019-09-03',information_date='2019-08-30')
  b=company_snapshot(facts,'A','2019-09-04',information_date='2019-09-03')
  self.assertTrue(np.isnan(a['log_assets']))
  self.assertAlmostEqual(b['log_assets'],np.log(100))



class RealDataTemporalIsolationTests(unittest.TestCase):
 def test_future_and_unmatured_labels_do_not_change_earlier_forecast(self):
  from pathlib import Path
  import pickle
  from portfolio_research.features import forecast_risk,FEATURE_SETS
  from portfolio_research.engine import ROOT,make_rebalance_dates
  path=ROOT/'artifacts/cache/context.pkl'
  if not path.exists():self.skipTest('Build the real-data context before running integration tests.')
  c=pickle.loads(path.read_bytes());panel=c['panel'];dates=make_rebalance_dates(c['splits']['validation'],'M')
  dates=dates[dates.year==2018][:1];cutoff=dates[0]
  altered=panel.copy();mask=altered.label_end>=cutoff
  altered.loc[mask,'target_log_risk_ratio']=1000.
  future=altered.date>cutoff
  altered.loc[future,FEATURE_SETS['full']]=9999.
  a,audit,_=forecast_risk(panel,dates,iterations=40)
  b,_,_=forecast_risk(altered,dates,iterations=40)
  self.assertTrue(np.allclose(a.predicted_downside_variance,b.predicted_downside_variance,atol=0,rtol=0))
  self.assertTrue((audit.last_training_label_end<audit.fit_date).all())
  self.assertTrue(((panel.date<cutoff)&(panel.label_end>=cutoff)).any())
 def test_all_real_filing_inputs_precede_actual_execution_close(self):
  from pathlib import Path
  import pickle
  from portfolio_research.engine import ROOT
  path=ROOT/'artifacts/cache/context.pkl'
  if not path.exists():self.skipTest('Build the real-data context before running integration tests.')
  c=pickle.loads(path.read_bytes());f=c['fundamentals'];mask=f.latest_used_available.notna()
  self.assertTrue((f.loc[mask,'latest_used_available']<=f.loc[mask,'information_date']).all())

if __name__=='__main__':unittest.main()
