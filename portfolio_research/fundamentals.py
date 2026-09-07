"""As-filed SEC features. No retrospective fiscal-period joins or backward filling."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

TAGS = {
 'assets': ['Assets'], 'liabilities': ['Liabilities'],
 'equity': ['StockholdersEquity','StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'],
 'cash': ['CashAndCashEquivalentsAtCarryingValue'],
 'current_assets': ['AssetsCurrent'], 'current_liabilities': ['LiabilitiesCurrent'],
 'revenue': ['RevenueFromContractWithCustomerExcludingAssessedTax','Revenues','SalesRevenueNet','RevenueFromContractWithCustomerIncludingAssessedTax'],
 'income': ['NetIncomeLoss','ProfitLoss'], 'operating_income': ['OperatingIncomeLoss'],
 'cfo': ['NetCashProvidedByUsedInOperatingActivities'],
 'capex': ['PaymentsToAcquirePropertyPlantAndEquipment'],
}
FLOWS = {'revenue','income','operating_income','cfo','capex'}
# Predecessor facts stop at the legal reorganization. They are never used as later subsidiary accounts.
LINEAGE = {
 'APA': (6769, '2021-03-02', 'https://www.sec.gov/Archives/edgar/data/1841666/000119312521063695/d127090d8k12b.htm'),
 'BLK': (1364742, '2024-10-01', 'https://www.sec.gov/Archives/edgar/data/1364742/000119312524229654/d856279d8k.htm'),
}
DOMAINS = {
 'profitability': {'roa':1, 'operating_margin':1},
 'balance_sheet': {'equity_to_assets':1, 'cash_to_assets':1, 'liabilities_to_assets':-1},
 'cash_generation': {'cfo_to_assets':1, 'accruals_to_assets':-1, 'fcf_to_assets':1},
 'liquidity': {'current_ratio':1},
}
RATIOS = list(dict.fromkeys(k for d in DOMAINS.values() for k in d))

def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()

def read_sec_facts(data_dir: Path, reference_file: Path):
 """Read only the mapped issuers, retaining filing date, accession, concept and duration."""
 mapping=json.loads(reference_file.read_text()); rows=[]; sources=[]
 for m in mapping:
  ticker=m['ticker']; identities=[(m['cik'],None)]
  if ticker in LINEAGE: identities.insert(0,LINEAGE[ticker][:2])
  for cik,cutoff in identities:
   p=data_dir/'company_raw/companyfacts'/f'CIK{cik:010d}.json'
   if not p.exists(): raise FileNotFoundError(p)
   obj=json.loads(p.read_text())
   if int(obj['cik'])!=cik:raise ValueError(f'Issuer identity mismatch: {p.name}')
   sources.append({'ticker':ticker,'cik':cik,'entity_name':obj['entityName'],'file':str(p.relative_to(data_dir)), 'sha256':sha256(p),'filing_cutoff_exclusive':cutoff})
   gaap=obj.get('facts',{}).get('us-gaap',{})
   for metric,tags in TAGS.items():
    for priority,tag in enumerate(tags):
     for r in gaap.get(tag,{}).get('units',{}).get('USD',[]):
      if r.get('form') not in ('10-K','10-K/A','10-Q','10-Q/A'): continue
      if not all(k in r for k in ('end','filed','val','accn')): continue
      if cutoff and r['filed']>=cutoff: continue
      rows.append({'ticker':ticker,'cik':cik,'metric':metric,'concept':tag,'priority':priority,'start':r.get('start'),'end':r['end'],'filed':r['filed'],'value':r['val'],'accn':r['accn'],'form':r['form']})
 f=pd.DataFrame(rows)
 for c in ('start','end','filed'): f[c]=pd.to_datetime(f[c])
 f['duration']=(f['end']-f['start']).dt.days
 valid=np.isfinite(f['value']) & (f['end']<=f['filed'])
 valid &= (~f['metric'].isin(FLOWS)) | (f['duration'].between(330,400) & f['form'].isin(['10-K','10-K/A']))
 invalid_count=int((~valid).sum()); f=f.loc[valid].copy()
 # Calendar-day filing timestamps do not establish intraday availability.
 # One business-day embargo is applied; the portfolio join is additionally strictly before formation.
 f['available']=f['filed']+pd.offsets.BDay(1)
 keys=['ticker','cik','concept','start','end','filed','accn']
 f=f.drop_duplicates(keys+['value'])
 conflicts=f.duplicated(keys,keep=False)
 conflict_count=int(conflicts.sum()); f=f.loc[~conflicts].copy()
 return f.sort_values(['ticker','available','end','priority']),pd.DataFrame(sources),{'invalid_or_nonannual_flow_rows':invalid_count,'ambiguous_fact_rows_excluded':conflict_count}

def _pick(known, metric, end=None, max_age_date=None):
 q=known[known.metric.eq(metric)]
 if end is not None: q=q[q.end.eq(end)]
 if max_age_date is not None: q=q[q.end>=max_age_date]
 if q.empty:return None
 # Latest economic period first, preferred equivalent concept next, latest known vintage last.
 return q.sort_values(['end','priority','filed','accn'],ascending=[False,True,False,False]).iloc[0]

def company_snapshot(facts, ticker, date, max_age_days=550, information_date=None):
 date=pd.Timestamp(date)
 cutoff=pd.Timestamp(information_date) if information_date is not None else date-pd.Timedelta(days=1)
 if pd.isna(cutoff) or pd.isna(date) or cutoff>=date:
  raise ValueError('information_date must strictly precede formation date')
 known=facts[(facts.ticker==ticker)&(facts.available<=cutoff)]
 out={'date':date,'ticker':ticker,'information_date':cutoff}; used=[]
 if known.empty:
  return {**out, **{x:np.nan for x in RATIOS},'revenue_growth':np.nan,'log_assets':np.nan,'report_age_days':np.nan,'filing_lag_days':np.nan,'annual_age_days':np.nan,'amended_fraction':np.nan,'known_filings':0,'fundamental_missing_fraction':1.,'latest_used_available':pd.NaT,'latest_used_filed':pd.NaT}
 oldest=date-pd.Timedelta(days=max_age_days)
 annual=known[known.metric.isin(['revenue','income','cfo']) & (known.end>=oldest)]
 annual_end=annual.end.max() if len(annual) else None
 def val(metric,end=None):
  r=_pick(known,metric,end=end,max_age_date=oldest)
  if r is None:return np.nan
  used.append(r);return float(r.value)
 def div(a,b):return a/b if np.isfinite(a) and np.isfinite(b) and b>0 else np.nan
 # Annual flows use matched fiscal-year-end denominators, not today's balance sheet.
 if annual_end is not None:
  a=val('assets',annual_end); ni=val('income',annual_end); cfo=val('cfo',annual_end)
  rev=val('revenue',annual_end); op=val('operating_income',annual_end); capex=val('capex',annual_end)
  out.update(roa=div(ni,a),operating_margin=div(op,rev),cfo_to_assets=div(cfo,a),accruals_to_assets=div(ni-cfo,a),fcf_to_assets=div(cfo-capex,a))
  prev=known[known.metric.eq('revenue') & ((annual_end-known.end).dt.days.between(330,400))]
  p=_pick(prev,'revenue');out['revenue_growth']=np.nan
  if p is not None and p.value>0:
   used.append(p);out['revenue_growth']=div(rev,float(p.value))-1
  out['annual_age_days']=(date-annual_end).days
 else:
  out.update({x:np.nan for x in ['roa','operating_margin','cfo_to_assets','accruals_to_assets','fcf_to_assets','revenue_growth','annual_age_days']})
 balance=_pick(known,'assets',max_age_date=oldest)
 if balance is not None:
  end=balance.end; a=val('assets',end)
  out.update(equity_to_assets=div(val('equity',end),a),cash_to_assets=div(val('cash',end),a),liabilities_to_assets=div(val('liabilities',end),a),current_ratio=div(val('current_assets',end),val('current_liabilities',end)),log_assets=np.log(a) if a>0 else np.nan)
 else:out.update({x:np.nan for x in ['equity_to_assets','cash_to_assets','liabilities_to_assets','current_ratio','log_assets']})
 if used:
  u=pd.DataFrame(used); latest=u.sort_values('filed').iloc[-1]
  out.update(report_age_days=(date-latest.filed).days,filing_lag_days=(latest.filed-latest.end).days,latest_used_available=u.available.max(),latest_used_filed=u.filed.max())
 else:out.update(report_age_days=np.nan,filing_lag_days=np.nan,latest_used_available=pd.NaT,latest_used_filed=pd.NaT)
 filings=known[['filed','accn','form']].drop_duplicates().tail(20)
 out['amended_fraction']=filings.form.str.endswith('/A').mean()
 out['known_filings']=known.accn.nunique()
 out['fundamental_missing_fraction']=np.mean([not np.isfinite(out[x]) for x in RATIOS])
 return out

def quality_scores(snapshot):
 """Four equally weighted economic domains; missing evidence is neutral, never good."""
 s=snapshot.copy();domain_values=[]
 for domain,components in DOMAINS.items():
  ranks=[]
  for col,direction in components.items():
   v=s[col].replace([np.inf,-np.inf],np.nan)
   # Ranks bound outliers without parameters estimated on future periods.
   ranks.append(((v.rank(method='average',pct=True)-0.5)*direction+0.5).where(v.notna(),0.5))
  s['quality_'+domain]=pd.concat(ranks,axis=1).mean(axis=1)
  domain_values.append(s['quality_'+domain])
 s['quality_score']=pd.concat(domain_values,axis=1).mean(axis=1).clip(0,1)
 s['quality_coverage']=1-s['fundamental_missing_fraction']
 return s

def build_fundamental_panel(facts, dates, tickers, trading_dates):
 blocks=[]
 by_ticker={t:g for t,g in facts.groupby("ticker")}
 calendar=pd.DatetimeIndex(trading_dates)
 if calendar.hasnans or not calendar.is_unique or not calendar.is_monotonic_increasing:
  raise ValueError('Trading calendar must be unique and chronological')
 for i,date in enumerate(dates):
  position=calendar.get_loc(date)
  if position==0:raise ValueError("No prior execution close")
  information_date=calendar[position-1]
  block=pd.DataFrame([company_snapshot(by_ticker.get(t,facts.iloc[:0]),t,date,information_date=information_date) for t in tickers]).set_index('ticker')
  blocks.append(quality_scores(block).reset_index())
  if (i+1)%36==0:print(f'Fundamental snapshots: {i+1}/{len(dates)}',flush=True)
 panel=pd.concat(blocks,ignore_index=True)
 used=panel.latest_used_available.notna()
 assert (panel.loc[used,'latest_used_available']<=panel.loc[used,'information_date']).all()
 assert (panel.information_date<panel.date).all()
 assert not panel.duplicated(['date','ticker']).any()
 return panel
