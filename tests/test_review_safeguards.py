"""Offline regression tests for the September 2026 repository review."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import pandas as pd

from portfolio_research.engine import (BacktestConfig, ROOT, export_result,
    panel_fingerprint, paired_bootstrap, run_backtest, validate_result)
from portfolio_research.workflow import verify_saved_result, verify_frozen, CONFIG
from portfolio_research.regimes import RegimeConfig
from portfolio_research.features import FEATURE_SETS


class EngineSafeguards(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range('2020-01-02', periods=3, freq='B')
        self.returns = pd.DataFrame({'A': [.1, -.01, .02], 'B': [0., .03, -.01]}, index=self.dates)
        self.market = {self.dates[0]: {'assets': ['A', 'B'], 'baseline': np.eye(2),
            'regime': np.eye(2), 'diagnostics': {'history_end': pd.Timestamp('2019-12-31')}}}
        self.config = BacktestConfig(max_weight=1.)

    def run_result(self, **overrides):
        args = dict(returns=self.returns, dates=self.dates, market=self.market,
                    config=self.config, strategy='equal_weight')
        args.update(overrides)
        return run_backtest(**args)

    def test_unknown_strategy_and_invalid_dates_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown strategy'):
            self.run_result(strategy='pipline1')
        for dates in [self.dates[::-1], self.dates.append(self.dates[:1])]:
            with self.assertRaisesRegex(ValueError, 'chronological'):
                self.run_result(dates=dates)

    def test_invalid_returns_and_penalties_rejected(self):
        for value in [np.inf, -1.1]:
            r = self.returns.copy(); r.iloc[1, 0] = value
            with self.assertRaisesRegex(ValueError, 'held-security'):
                self.run_result(returns=r)
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            self.run_result(quality_strength=-1)

    def test_absent_quality_is_not_reported_as_measured(self):
        result = self.run_result()
        self.assertTrue(result.diagnostics.portfolio_quality.isna().all())

    def test_cache_fingerprint_includes_asset_identity(self):
        self.assertNotEqual(panel_fingerprint(self.returns), panel_fingerprint(self.returns.rename(columns={'A': 'Z'})))
        self.assertNotEqual(panel_fingerprint(self.returns), panel_fingerprint(self.returns.iloc[:, ::-1]))

    def test_invalid_saved_accounting_is_rejected(self):
        result = self.run_result()
        result.returns.iloc[1, result.returns.columns.get_loc('net_return')] += .01
        with self.assertRaisesRegex(ValueError, 'accounting'):
            validate_result(result)

    def test_notebook_verification_does_not_overwrite_quality(self):
        calculated = self.run_result(); recorded = copy.deepcopy(calculated)
        recorded.diagnostics['portfolio_quality'] = .7
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            export_result(recorded, 'equal_weight', 'test', root)
            path = root/'artifacts/results/test/equal_weight/diagnostics.csv'
            before = path.read_bytes()
            verified = verify_saved_result(calculated, 'equal_weight', 'test', root)
            self.assertEqual(path.read_bytes(), before)
            self.assertAlmostEqual(verified.diagnostics.portfolio_quality.iloc[0], .7)
            calculated.returns.iloc[0, 0] += .01
            with self.assertRaisesRegex(ValueError, 'differ'):
                verify_saved_result(calculated, 'equal_weight', 'test', root)

    def test_bootstrap_rejects_unpaired_or_invalid_windows(self):
        a = self.returns.A
        for b, block in [(a.iloc[::-1], 1), (a, 4), (a, 0)]:
            with self.assertRaises(ValueError):
                paired_bootstrap(a, b, replications=10, block=block)


class ReproductionSafeguards(unittest.TestCase):
    def test_synthetic_catboost_isolation_from_future_and_unmatured_labels(self):
        from portfolio_research.features import forecast_risk
        rng = np.random.default_rng(7)
        dates = pd.date_range('2010-01-01', periods=36, freq='MS')
        panel = pd.MultiIndex.from_product([dates, [f'T{i:02}' for i in range(50)]], names=['date','ticker']).to_frame(index=False)
        for feature in FEATURE_SETS['full']:
            panel[feature] = rng.normal(size=len(panel))
        panel['downside_63'] = .0001
        panel['label_end'] = panel.date + pd.offsets.BDay(21)
        panel['target_log_risk_ratio'] = rng.normal(size=len(panel))
        panel['future_downside_variance'] = .0001*np.exp(panel.target_log_risk_ratio)
        cutoff = dates[30]
        changed = panel.copy()
        changed.loc[changed.label_end>=cutoff, 'target_log_risk_ratio'] = 1000.
        changed.loc[changed.date>cutoff, FEATURE_SETS['full']] = 9999.
        a, audit, _ = forecast_risk(panel, [cutoff], iterations=5)
        b, _, _ = forecast_risk(changed, [cutoff], iterations=5)
        np.testing.assert_array_equal(a.predicted_downside_variance, b.predicted_downside_variance)
        self.assertTrue((audit.last_training_label_end<cutoff).all())
        self.assertTrue(((panel.date<cutoff)&(panel.label_end>=cutoff)).any())

    def test_synthetic_hmm_and_covariance_ignore_future_features(self):
        from portfolio_research.regimes import estimate_regime_covariance
        rng = np.random.default_rng(19)
        dates = pd.date_range('2019-01-01', periods=120, freq='B')
        features = pd.DataFrame(rng.normal(size=(120,4)), index=dates,
            columns=['market_return','realized_volatility','drawdown','average_correlation'])
        history = pd.DataFrame(rng.normal(0,.01,size=(60,3)), index=dates[40:100], columns=['A','B','C'])
        cfg = RegimeConfig(hmm_lookback=100,min_hmm_observations=80,random_restarts=2,max_iterations=50)
        changed = features.copy(); changed.loc[dates[100]:] = 1e6
        a, diag = estimate_regime_covariance(history, features, dates[100], cfg)
        b, _ = estimate_regime_covariance(history, changed, dates[100], cfg)
        # Threaded BLAS may differ at the last floating-point bits between fits.
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-15)
        self.assertTrue((np.linalg.eigvalsh(a)>0).all())
        self.assertLess(diag['hmm_history_end'], dates[100])

    def test_frozen_settings_and_required_sources(self):
        c = {'hashes': {'train': 'same'}, 'fundamental_key': 'f', 'market_key': 'm',
             'splits': {'test': pd.date_range('2021-01-04', periods=2)}, 'config': CONFIG}
        spec = {'input_hashes': c['hashes'], 'code_hashes': {'code': 'same'},
                'selection_split': '2017-2020', 'baseline': asdict(CONFIG),
                'regime': asdict(RegimeConfig()), 'features': FEATURE_SETS,
                'fundamental_key': 'f', 'market_key': 'm',
                'test_range': ['2021-01-04', '2021-01-05'], 'ml': {'feature_set': 'full'}}
        with patch('portfolio_research.workflow.code_hashes', return_value={'code': 'same'}):
            verify_frozen(c, spec)
            with self.assertRaisesRegex(ValueError, 'OHLCV'):
                verify_frozen(c, spec, require_features=True)
            for key in ['baseline', 'regime', 'input_hashes', 'features', 'test_range']:
                changed = copy.deepcopy(spec); changed[key] = None
                with self.subTest(key=key), self.assertRaises(ValueError):
                    verify_frozen(c, changed)

    def test_failed_notebook_invalidates_previous_success(self):
        import nbformat
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root/'scripts').mkdir(); (root/'artifacts/cache').mkdir(parents=True)
            shutil.copy2(ROOT/'scripts/execute_notebooks.py', root/'scripts/execute_notebooks.py')
            nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('raise RuntimeError("expected review failure")')]), root/'failure.ipynb')
            (root/'artifacts/run_manifest.json').write_text(json.dumps({'status': 'completed'}))
            record = root/'artifacts/cache/failure.ipynb.execution.json'
            record.write_text(json.dumps({'status': 'completed'}))
            result = subprocess.run([sys.executable, str(root/'scripts/execute_notebooks.py'), '--single', 'failure.ipynb'], capture_output=True, text=True, timeout=60)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(json.loads(record.read_text())['status'], 'failed')
            self.assertEqual(json.loads((root/'artifacts/run_manifest.json').read_text())['status'], 'failed')
            outputs = nbformat.read(root/'failure.ipynb', as_version=4).cells[0].outputs
            self.assertTrue(any(o.output_type == 'error' for o in outputs))

    def test_failed_experiment_invalidates_previous_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'scripts').mkdir(); (root/'artifacts').mkdir()
            shutil.copy2(ROOT/'scripts/run_experiments.py',root/'scripts/run_experiments.py')
            shutil.copytree(ROOT/'portfolio_research',root/'portfolio_research',ignore=shutil.ignore_patterns('__pycache__'))
            path=root/'artifacts/run_manifest.json'
            path.write_text(json.dumps({'status':'completed'}))
            attempt=subprocess.run([sys.executable,str(root/'scripts/run_experiments.py')],capture_output=True,text=True,timeout=60)
            self.assertNotEqual(attempt.returncode,0)
            record=json.loads(path.read_text())
            self.assertEqual(record['status'],'failed')
            self.assertEqual(record['failure_stage'],'experiments')
            self.assertIn('FileNotFoundError',record['error'])


if __name__ == '__main__':
    unittest.main()
