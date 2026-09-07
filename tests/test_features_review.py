"""Calendar and missing-report edge cases that do not need private raw inputs."""
import unittest

import numpy as np
import pandas as pd

from portfolio_research.features import FEATURE_SETS, FUNDAMENTAL_FEATURES, make_feature_panel
from portfolio_research.fundamentals import RATIOS, build_fundamental_panel, company_snapshot


def sample_facts():
    return pd.DataFrame([{
        'ticker': 'A', 'metric': 'assets', 'value': 100.,
        'end': pd.Timestamp('2019-12-31'), 'filed': pd.Timestamp('2020-02-03'),
        'available': pd.Timestamp('2020-02-04'), 'priority': 0,
        'accn': 'first', 'form': '10-K',
    }])


class FundamentalReviewTests(unittest.TestCase):
    def test_information_cutoff_must_precede_formation(self):
        for cutoff in ['2020-02-05', '2020-02-06']:
            with self.subTest(cutoff=cutoff), self.assertRaisesRegex(ValueError, 'information_date'):
                company_snapshot(sample_facts(), 'A', '2020-02-05', information_date=cutoff)

    def test_issuer_without_supported_facts_gets_neutral_snapshot(self):
        calendar = pd.date_range('2020-02-03', periods=4, freq='B')
        panel = build_fundamental_panel(sample_facts(), [calendar[-1]], ['A', 'B'], calendar)
        absent = panel.set_index('ticker').loc['B']
        self.assertEqual(absent.known_filings, 0)
        self.assertEqual(absent.quality_coverage, 0.)
        self.assertEqual(absent.quality_score, .5)
        self.assertTrue(absent[RATIOS].isna().all())
        self.assertTrue(pd.isna(absent.latest_used_available))

    def test_empty_but_typed_fact_panel_is_supported(self):
        calendar = pd.date_range('2020-02-03', periods=4, freq='B')
        panel = build_fundamental_panel(sample_facts().iloc[:0], [calendar[-1]], ['A', 'B'], calendar)
        self.assertTrue((panel.quality_score == .5).all())
        self.assertTrue((panel.quality_coverage == 0.).all())

    def test_later_restatement_changes_only_later_snapshot(self):
        facts = sample_facts()
        amended = facts.copy()
        amended['value'] = 120.
        amended['filed'] = pd.Timestamp('2020-02-20')
        amended['available'] = pd.Timestamp('2020-02-21')
        amended['accn'] = 'amended'
        amended['form'] = '10-K/A'
        both = pd.concat([facts, amended], ignore_index=True)
        before = company_snapshot(both, 'A', '2020-02-05')
        after = company_snapshot(both, 'A', '2020-02-24')
        self.assertEqual(before['log_assets'], np.log(100.))
        self.assertEqual(after['log_assets'], np.log(120.))


class FeatureReviewTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range('2018-01-01', periods=300, freq='B')
        phase = np.arange(300, dtype=float)
        asset_returns = np.column_stack([.001 + .012 * np.sin(phase), .0005 + .008 * np.cos(phase)])
        self.prices = pd.DataFrame(100 * np.cumprod(1 + asset_returns, axis=0),
                                   index=self.dates, columns=['A', 'B'])
        self.returns = self.prices.pct_change(fill_method=None)
        self.date = self.dates[-25]
        self.info_date = self.dates[-26]
        self.market = {self.date: {
            'assets': ['A', 'B'], 'baseline': np.diag([.0001, .0002]),
            'regime': np.diag([.0002, .0003]),
            'diagnostics': {'forecast_stress_probability': .4},
        }}
        self.regime_features = pd.DataFrame({
            'realized_volatility': .2, 'drawdown': -.05, 'average_correlation': .3,
        }, index=self.dates)
        self.fundamentals = pd.DataFrame({
            'date': [self.date] * 2, 'ticker': ['A', 'B'],
            'information_date': [self.info_date] * 2,
            'latest_used_available': [pd.NaT] * 2,
            **{c: [0., 0.] for c in FUNDAMENTAL_FEATURES + ['report_age_days', 'filing_lag_days', 'annual_age_days', 'amended_fraction', 'known_filings', 'fundamental_missing_fraction']},
        })
        self.ohlcv = {'Close': self.prices.copy(), 'High': self.prices * 1.01,
                      'Low': self.prices * .99, 'Volume': self.prices * 0 + 1_000_000}

    def make_panel(self, **overrides):
        values = dict(prices=self.prices, returns=self.returns, fundamentals=self.fundamentals,
                      market=self.market, regime_features=self.regime_features, ohlcv=self.ohlcv)
        values.update(overrides)
        return make_feature_panel(**values)

    def test_misaligned_ohlcv_calendar_is_rejected(self):
        shifted = {key: value.set_axis(value.index + pd.offsets.BDay(1))
                   for key, value in self.ohlcv.items()}
        with self.assertRaisesRegex(ValueError, 'calendar|index|align'):
            self.make_panel(ohlcv=shifted)

    def test_misaligned_price_calendar_is_rejected(self):
        shifted = self.prices.set_axis(self.prices.index + pd.offsets.BDay(1))
        with self.assertRaisesRegex(ValueError, 'calendar|index|align'):
            self.make_panel(prices=shifted)

    def test_future_fundamental_snapshot_is_rejected(self):
        future = self.fundamentals.copy()
        future['information_date'] = self.date
        with self.assertRaisesRegex(ValueError, 'fundamental|information'):
            self.make_panel(fundamentals=future)

    def test_missing_fundamental_snapshot_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'fundamental|snapshot'):
            self.make_panel(fundamentals=self.fundamentals.iloc[:1])

    def test_future_prices_and_activity_do_not_change_current_features(self):
        original = self.make_panel()
        prices = self.prices.copy()
        prices.loc[self.date:] *= 3.
        returns = prices.pct_change(fill_method=None)
        ohlcv = {key: value.copy() for key, value in self.ohlcv.items()}
        for value in ohlcv.values():
            value.loc[self.date:] *= 7.
        changed = self.make_panel(prices=prices, returns=returns, ohlcv=ohlcv)
        pd.testing.assert_frame_equal(original[FEATURE_SETS['full']], changed[FEATURE_SETS['full']])
        self.assertFalse(np.allclose(original.future_downside_variance,
                                     changed.future_downside_variance))

    def test_labels_include_formation_session_and_only_twenty_one_sessions(self):
        panel = self.make_panel().set_index('ticker')
        position = self.returns.index.get_loc(self.date)
        expected = np.minimum(self.returns.iloc[position:position + 21], 0).pow(2).mean().clip(lower=1e-10)
        np.testing.assert_allclose(panel.future_downside_variance, expected)
        self.assertTrue((panel.label_end == self.returns.index[position + 20]).all())
        self.assertTrue((panel.price_information_end == self.info_date).all())


if __name__ == '__main__':
    unittest.main()
