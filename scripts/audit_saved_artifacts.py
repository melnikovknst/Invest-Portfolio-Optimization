"""Audit retained evidence without raw data or silently rerunning the experiment.

Exit 1 on a failed check; warnings are explicitly reported, never called passes.
Use --bootstrap to independently regenerate all retained bootstrap intervals.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from portfolio_research.engine import load_result, validate_result, metrics, LABELS, risk_score, paired_bootstrap
from portfolio_research.features import prediction_diagnostics
from portfolio_research.fundamentals import sha256
from portfolio_research.presentation import annual_metrics, cost_sensitivity, regime_conditioned

ORIGINAL_COMMIT = '93676de158c5df4a6cc0cee3427707869f39dbd5'


def audit(bootstrap=False):
    report = {'original_experiment_commit': ORIGINAL_COMMIT,
              'scope': 'Saved-evidence reconciliation, not a raw-data reproduction',
              'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
              'checks': [], 'warnings': [], 'failures': []}
    def check(name, fn):
        try:
            detail = fn()
            report['checks'].append({'name': name, 'status': 'passed', 'detail': detail})
        except Exception as exc:
            report['failures'].append({'name': name, 'error': str(exc)})
    def close(a, b):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9, equal_nan=True)
    all_results = {}
    errors = []
    for path in sorted((ROOT/'artifacts/results').glob('*/*/metrics.csv')):
        split, name = path.parts[-3:-1]
        def result_check(split=split, name=name, path=path):
            r = load_result(name, split); validate_result(r)
            actual = metrics(r)
            saved = pd.read_csv(path, index_col=0).value.reindex(actual.index)
            close(actual, saved)
            errors.append(float((actual-saved).abs().max()))
            w, d = r.target_weights, r.diagnostics
            close(w.pow(2).sum(axis=1), d.hhi)
            close(1/w.pow(2).sum(axis=1), d.effective_assets)
            close(w.max(axis=1), d.maximum_weight)
            close(r.returns.loc[d.index, 'turnover'], d.turnover)
            if not d.solver_success.all(): raise ValueError('Solver failure recorded')
            all_results[(split, name)] = r
            return {'days': len(r.returns), 'rebalances': len(w)}
        check(f'result/{split}/{name}', result_check)
    report['maximum_metric_absolute_error'] = max(errors, default=None)
    report['result_folders_checked'] = len(errors)
    for split in ['validation', 'test']:
        summary = pd.read_csv(ROOT/f'artifacts/tables/{split}_metrics.csv', index_col=0)
        for name in LABELS:
            def summary_check(split=split, name=name, summary=summary):
                actual = metrics(all_results[(split, name)])
                cols = actual.index.drop('Average Portfolio Quality')
                close(actual[cols], summary.loc[name, cols])
                aq, sq = actual['Average Portfolio Quality'], summary.loc[name, 'Average Portfolio Quality']
                if not np.isclose(aq, sq, atol=1e-9, rtol=0, equal_nan=True):
                    report['warnings'].append({'type': 'quality_exposure_mismatch', 'split': split, 'strategy': name,
                        'diagnostic_average': aq, 'summary_average': sq,
                        'resolution': 'Original evidence retained; requires source SEC snapshots to reconstruct measured exposure.'})
            check(f'summary/{split}/{name}', summary_check)
    for split in ['validation', 'test']:
        def forecasts(split=split):
            p = pd.read_csv(ROOT/f'artifacts/tables/pipeline3_{split}_predictions.csv', parse_dates=['date','label_end','model_training_cutoff'])
            if p.duplicated(['date','ticker']).any(): raise ValueError('Duplicate predictions')
            if not (p.model_training_cutoff<=p.date).all(): raise ValueError('Prediction precedes fit')
            close(p.predicted_downside_variance, p.downside_63*np.exp(p.predicted_log_risk_ratio))
            t = pd.read_csv(ROOT/f'artifacts/tables/pipeline3_{split}_training_audit.csv', parse_dates=['fit_date','last_training_formation','last_training_label_end'])
            if not ((t.last_training_formation<t.fit_date)&(t.last_training_label_end<t.fit_date)).all():
                raise ValueError('Training cutoff violation')
            if split=='test':
                expected = pd.read_csv(ROOT/'artifacts/tables/test_prediction_metrics.csv', index_col=0)
                close(prediction_diagnostics(p).loc[expected.index,expected.columns], expected)
            return {'predictions': len(p), 'complete_labels': int(p.label_end.notna().sum()), 'annual_fits': len(t)}
        check(f'forecasts/{split}', forecasts)
    def ablations():
        expected = pd.read_csv(ROOT/'artifacts/tables/test_forecast_feature_ablation.csv', index_col=0)
        for name in ['price_regime', 'plus_fundamentals', 'full']:
            filename = 'pipeline3_test_predictions.csv' if name=='full' else f'ablation_{name}_test_predictions.csv'
            pred = pd.read_csv(ROOT/'artifacts/tables'/filename)
            actual = prediction_diagnostics(pred).loc['CatBoost']
            close(actual[expected.columns], expected.loc[name])
    check('forecast_feature_ablations', ablations)
    def selection():
        spec = json.loads((ROOT/'artifacts/frozen_spec.json').read_text())
        for filename, tie, expected in [
            ('pipeline2_validation_grid.csv', ['quality_strength'], {'quality_strength': spec['quality_strength']}),
            ('pipeline3_validation_grid.csv', ['ml_strength','depth'], {'ml_strength': spec['ml']['ml_strength'], 'depth': spec['ml']['depth']})]:
            table = pd.read_csv(ROOT/'artifacts/tables'/filename, index_col=0)
            close(risk_score(table), table.risk_score)
            chosen = table.sort_values(['risk_score']+tie, kind='stable').iloc[0]
            for k, v in expected.items(): close(chosen[k], v)
    check('validation_selection', selection)
    primary = {name: all_results[('test', name)] for name in LABELS if ('test', name) in all_results}
    for filename, fn, keys in [
        ('test_annual_metrics.csv', annual_metrics, ['strategy','year']),
        ('test_cost_sensitivity.csv', cost_sensitivity, ['strategy','cost_bps']),
        ('test_regime_conditioned.csv', regime_conditioned, ['strategy','regime'])]:
        def derived(filename=filename, fn=fn, keys=keys):
            actual = fn(primary).set_index(keys)
            expected = pd.read_csv(ROOT/'artifacts/tables'/filename).set_index(keys)
            cols = expected.select_dtypes(include='number').columns.drop('Average Portfolio Quality', errors='ignore')
            close(actual.loc[expected.index,cols], expected[cols])
        check(filename, derived)
    if bootstrap:
        def intervals():
            saved = pd.read_csv(ROOT/'artifacts/tables/test_paired_bootstrap.csv')
            for (a,b,block), group in saved.groupby(['reference','experiment','block_length']):
                actual = paired_bootstrap(primary[a].returns.net_return, primary[b].returns.net_return, block=int(block))
                expected = group.set_index('metric').loc[actual.index,actual.columns]
                close(actual, expected)
            return {'comparisons_and_block_lengths': saved.groupby(['reference','experiment','block_length']).ngroups, 'replications_each': 2000}
        check('bootstrap_intervals', intervals)
    def historical_code():
        spec = json.loads((ROOT/'artifacts/frozen_spec.json').read_text())
        manifest = json.loads((ROOT/'artifacts/run_manifest.json').read_text())
        if manifest['frozen_spec_sha256']!=sha256(ROOT/'artifacts/frozen_spec.json'): raise ValueError('Frozen specification hash changed')
        for name, digest in spec['code_hashes'].items():
            data = subprocess.check_output(['git','show',f'{ORIGINAL_COMMIT}:portfolio_research/{name}'], cwd=ROOT)
            if hashlib.sha256(data).hexdigest()!=digest: raise ValueError(f'Original code hash differs: {name}')
        report['current_code_sha256'] = {name: sha256(ROOT/'portfolio_research'/name) for name in spec['code_hashes']}
        report['current_code_matches_original_run'] = report['current_code_sha256']==spec['code_hashes']
    check('historical_code_provenance', historical_code)
    def notebooks():
        output = []
        for name in ['baseline.ipynb','pipeline1_regime_aware_gmv.ipynb','pipeline2_fundamental_quality_gmv.ipynb','pipeline3_catboost_risk_gmv.ipynb','results_comparison.ipynb']:
            nb = json.loads((ROOT/name).read_text())
            original = json.loads(subprocess.check_output(['git','show',f'{ORIGINAL_COMMIT}:{name}'], cwd=ROOT))
            cells = [c for c in nb['cells'] if c['cell_type']=='code']
            old = [c for c in original['cells'] if c['cell_type']=='code']
            if [c.get('outputs') for c in cells]!=[c.get('outputs') for c in old]: raise ValueError(f'Historical outputs changed: {name}')
            outputs = [o for c in cells for o in c.get('outputs', [])]
            if any(o['output_type']=='error' for o in outputs): raise ValueError(f'Notebook error: {name}')
            if any(c.get('execution_count') is None for c in cells): raise ValueError(f'Unexecuted original cell: {name}')
            for cell in cells: compile(''.join(cell['source']), name, 'exec')
            output.append({'name': name, 'code_cells': len(cells), 'figures': sum('image/png' in o.get('data',{}) for o in outputs), 'html_outputs': sum('text/html' in o.get('data',{}) for o in outputs)})
        return output
    check('historical_notebook_outputs_preserved_and_current_sources_compile', notebooks)
    def article():
        with zipfile.ZipFile(ROOT/'Portfolio optimization.docx') as z:
            xml = ET.fromstring(z.read('word/document.xml'))
        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
              'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
        def cell_text(node):
            if node.tag in (f"{{{ns['w']}}}t", f"{{{ns['m']}}}t"):
                return (node.text or '').replace('−','-')
            if node.tag==f"{{{ns['m']}}}sSup":
                return cell_text(node.find('m:e',ns))+'^'+cell_text(node.find('m:sup',ns))
            return ''.join(cell_text(child) for child in node)
        actual = []
        for table in xml.findall('.//w:tbl', ns):
            actual.append([[cell_text(cell) for cell in row.findall('w:tc',ns)] for row in table.findall('w:tr',ns)])
        expected = []; current = []
        for line in (ROOT/'paper/manuscript.md').read_text().splitlines()+['']:
            if line.startswith('|'):
                row = [x.strip().replace('−','-') for x in line.strip('|').split('|')]
                if not all(x=='---' for x in row): current.append(row)
            elif current: expected.append(current); current=[]
        if actual!=expected: raise ValueError('DOCX tables differ from Markdown tables')
        manifest = json.loads((ROOT/'paper/article_manifest.json').read_text())
        if manifest['docx_sha256']!=sha256(ROOT/'Portfolio optimization.docx'): raise ValueError('DOCX hash mismatch')
        for name, digest in manifest['result_table_sha256'].items():
            if sha256(ROOT/'artifacts/tables'/name)!=digest: raise ValueError(f'Article source changed: {name}')
        for name, digest in manifest.get('presentation_source_sha256',{}).items():
            if sha256(ROOT/name)!=digest: raise ValueError(f'Presentation source changed: {name}')
        return {'tables': len(actual), 'docx_sha256': manifest['docx_sha256']}
    check('article_table_and_source_integrity', article)
    report['warnings'].append({'type': 'raw_data_unavailable', 'detail': 'No fresh HMM/SEC/CatBoost reproduction was performed; restore the hash-matched price splits, SEC issuer files and OHLCV source for a new validation freeze.'})
    report['status'] = 'failed' if report['failures'] else 'passed_with_disclosed_limitations'
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--bootstrap', action='store_true'); parser.add_argument('--output', type=Path)
    args = parser.parse_args(); result = audit(args.bootstrap)
    content = json.dumps(result, indent=2, default=str)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(content)
    print(content)
    raise SystemExit(bool(result['failures']))
