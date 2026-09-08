"""Fail-fast audit of the current frozen experiment and submission artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from portfolio_research.engine import LABELS, load_result, metrics, paired_bootstrap, risk_score, validate_result
from portfolio_research.features import prediction_diagnostics
from portfolio_research.presentation import annual_metrics, cost_sensitivity, regime_conditioned


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def audit(bootstrap: bool = False) -> dict:
    report = {
        "scope": "Current raw-source, frozen-experiment, notebook, and submission reconciliation",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "checks": [], "failures": [], "warnings": [],
    }

    def check(name, fn):
        try:
            report["checks"].append({"name": name, "status": "passed", "detail": fn()})
        except Exception as exc:
            report["failures"].append({"name": name, "error": str(exc)})

    def close(a, b, atol=1e-9):
        np.testing.assert_allclose(a, b, rtol=0, atol=atol, equal_nan=True)

    def sources():
        universe = json.loads((ROOT / "artifacts/tables/universe_audit.json").read_text())
        price = Path(universe["source_file"])
        if digest(price) != universe["source_sha256"]:
            raise ValueError("Raw OHLCV source hash mismatch")
        sec_zip = ROOT / "data/company_raw/companyfacts.zip"
        if digest(sec_zip) != "a62a7c6a9bfea0d721a73ac9439b8a07a089da1eeee01aa71ef06917e8be77eb":
            raise ValueError("SEC Company Facts archive hash mismatch")
        identity = ROOT / universe["sec_snapshot"]
        if digest(identity) != universe["sec_snapshot_sha256"]:
            raise ValueError("SEC identity snapshot hash mismatch")
        profile = pd.read_csv(ROOT / "artifacts/tables/universe_source_profile.csv")
        eligible = profile.loc[profile["eligible_complete_history"], "Ticker"]
        if len(eligible) != 323 or universe["eligible_tickers"] != 323:
            raise ValueError("Complete-history universe is not 323 tickers")
        source_rows = pd.read_csv(ROOT / "artifacts/tables/fundamental_sources.csv")
        for row in source_rows.itertuples(index=False):
            path = ROOT / "data" / row.file
            if digest(path) != row.sha256:
                raise ValueError(f"Company Facts file changed: {row.file}")
        return {"source_rows": universe["source_rows"], "eligible_tickers": 323, "issuer_files": len(source_rows)}

    check("raw_sources_and_universe", sources)

    def frozen():
        spec = json.loads((ROOT / "artifacts/frozen_spec.json").read_text())
        manifest = json.loads((ROOT / "artifacts/run_manifest.json").read_text())
        if manifest["frozen_spec_sha256"] != digest(ROOT / "artifacts/frozen_spec.json"):
            raise ValueError("Frozen specification hash mismatch")
        for name, expected in spec["code_hashes"].items():
            if digest(ROOT / "portfolio_research" / name) != expected:
                raise ValueError(f"Economic code differs from freeze: {name}")
        for name, expected in spec["input_hashes"].items():
            if digest(ROOT / "data" / name) != expected:
                raise ValueError(f"Price split differs from freeze: {name}")
        expected_selection = {"quality_strength": 0.5, "depth": 4, "ml_strength": 0.25, "iterations": 300}
        actual = {"quality_strength": spec["quality_strength"], **{k: spec["ml"][k] for k in ["depth", "ml_strength", "iterations"]}}
        if actual != expected_selection:
            raise ValueError(f"Unexpected frozen selection: {actual}")
        return actual

    check("frozen_code_inputs_and_selection", frozen)

    results = {}
    errors = []
    for path in sorted((ROOT / "artifacts/results").glob("*/*/metrics.csv")):
        split, name = path.parts[-3:-1]

        def result_check(split=split, name=name, path=path):
            result = load_result(name, split)
            validate_result(result)
            actual = metrics(result)
            saved = pd.read_csv(path, index_col=0).value.reindex(actual.index)
            close(actual, saved)
            errors.append(float((actual - saved).abs().max()))
            d, w = result.diagnostics, result.target_weights
            close(w.pow(2).sum(axis=1), d.hhi)
            close(1 / w.pow(2).sum(axis=1), d.effective_assets)
            close(w.max(axis=1), d.maximum_weight)
            if not d.solver_success.all():
                raise ValueError("Solver failure recorded")
            results[(split, name)] = result
            return {"days": len(result.returns), "rebalances": len(w)}

        check(f"result/{split}/{name}", result_check)
    report["maximum_metric_absolute_error"] = max(errors, default=None)

    for split in ["validation", "test"]:
        def summary_check(split=split):
            table = pd.read_csv(ROOT / f"artifacts/tables/{split}_metrics.csv", index_col=0)
            for name in LABELS:
                actual = metrics(results[(split, name)])
                close(actual, table.loc[name, actual.index])
            return {"strategies": len(LABELS), "rows": len(results[(split, "baseline")].returns)}
        check(f"summary/{split}", summary_check)

    def forecasts():
        for split in ["validation", "test"]:
            p = pd.read_csv(ROOT / f"artifacts/tables/pipeline3_{split}_predictions.csv", parse_dates=["date", "label_end", "model_training_cutoff"])
            if p.duplicated(["date", "ticker"]).any():
                raise ValueError(f"Duplicate {split} predictions")
            close(p.predicted_downside_variance, p.downside_63 * np.exp(p.predicted_log_risk_ratio))
            t = pd.read_csv(ROOT / f"artifacts/tables/pipeline3_{split}_training_audit.csv", parse_dates=["fit_date", "last_training_formation", "last_training_label_end"])
            if not ((t.last_training_formation < t.fit_date) & (t.last_training_label_end < t.fit_date)).all():
                raise ValueError(f"Training cutoff violation in {split}")
        expected = pd.read_csv(ROOT / "artifacts/tables/test_prediction_metrics.csv", index_col=0)
        actual = prediction_diagnostics(pd.read_csv(ROOT / "artifacts/tables/pipeline3_test_predictions.csv"))
        close(actual.loc[expected.index, expected.columns], expected)
        return {"test_rows": int(expected.loc["CatBoost", "rows"]), "test_months": int(expected.loc["CatBoost", "months"])}

    check("forecast_chronology_and_metrics", forecasts)

    def selection():
        spec = json.loads((ROOT / "artifacts/frozen_spec.json").read_text())
        q = pd.read_csv(ROOT / "artifacts/tables/pipeline2_validation_grid.csv", index_col=0)
        close(risk_score(q), q.risk_score)
        if float(q.sort_values(["risk_score", "quality_strength"]).iloc[0].quality_strength) != spec["quality_strength"]:
            raise ValueError("Quality selection does not reproduce")
        m = pd.read_csv(ROOT / "artifacts/tables/pipeline3_validation_grid.csv", index_col=0)
        close(risk_score(m), m.risk_score)
        chosen = m.sort_values(["risk_score", "ml_strength", "depth"]).iloc[0]
        if int(chosen.depth) != spec["ml"]["depth"] or float(chosen.ml_strength) != spec["ml"]["ml_strength"]:
            raise ValueError("ML selection does not reproduce")
        return {"quality_strength": spec["quality_strength"], "depth": int(chosen.depth), "ml_strength": float(chosen.ml_strength)}

    check("validation_selection", selection)
    primary = {name: results[("test", name)] for name in LABELS}
    for filename, fn, keys in [
        ("test_annual_metrics.csv", annual_metrics, ["strategy", "year"]),
        ("test_cost_sensitivity.csv", cost_sensitivity, ["strategy", "cost_bps"]),
        ("test_regime_conditioned.csv", regime_conditioned, ["strategy", "regime"]),
    ]:
        def derived(filename=filename, fn=fn, keys=keys):
            actual = fn(primary).set_index(keys)
            expected = pd.read_csv(ROOT / "artifacts/tables" / filename).set_index(keys)
            cols = expected.select_dtypes(include="number").columns
            close(actual.loc[expected.index, cols], expected[cols])
            return {"rows": len(expected)}
        check(filename, derived)

    if bootstrap:
        def intervals():
            saved = pd.read_csv(ROOT / "artifacts/tables/test_paired_bootstrap.csv")
            for (a, b, block), group in saved.groupby(["reference", "experiment", "block_length"]):
                actual = paired_bootstrap(primary[a].returns.net_return, primary[b].returns.net_return, block=int(block))
                close(actual, group.set_index("metric").loc[actual.index, actual.columns])
            return {"groups": saved.groupby(["reference", "experiment", "block_length"]).ngroups, "replications": 2000}
        check("bootstrap_intervals", intervals)

    def notebooks():
        manifest = json.loads((ROOT / "artifacts/run_manifest.json").read_text())
        executions = manifest.get("notebook_execution", [])
        if not executions or any(item.get("status") != "completed" for item in executions):
            raise ValueError("Notebook manifest is not completed")
        recorded = {item.get("notebook") for item in executions}
        output = []
        for name in ["baseline.ipynb", "pipeline1_regime_aware_gmv.ipynb", "pipeline2_fundamental_quality_gmv.ipynb", "pipeline3_catboost_risk_gmv.ipynb", "results_comparison.ipynb"]:
            if name not in recorded:
                raise ValueError(f"Notebook is absent from execution manifest: {name}")
            nb = json.loads((ROOT / name).read_text())
            cells = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
            outputs = [o for cell in cells for o in cell.get("outputs", [])]
            if any(o["output_type"] == "error" for o in outputs) or any(cell.get("execution_count") is None for cell in cells):
                raise ValueError(f"Notebook incomplete or failed: {name}")
            for cell in cells:
                compile("".join(cell["source"]), name, "exec")
            output.append({"name": name, "code_cells": len(cells), "outputs": len(outputs)})
        return output

    check("executed_notebooks", notebooks)

    def article():
        md = (ROOT / "paper/manuscript.md").read_text()
        chars = len(md)
        if chars > 40000:
            raise ValueError(f"Manuscript exceeds 40,000 characters: {chars}")
        abstract = md.split("## Abstract", 1)[1].split("Keywords:", 1)[0]
        words = len(re.findall(r"\b[\w–-]+\b", abstract))
        if not 200 <= words <= 250:
            raise ValueError(f"Abstract has {words} words")
        with zipfile.ZipFile(ROOT / "Portfolio_optimization.docx") as archive:
            xml = ET.fromstring(archive.read("word/document.xml"))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main", "m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}
        if len(xml.findall(".//m:oMath", ns)) < 12:
            raise ValueError("DOCX equations missing")
        manifest = json.loads((ROOT / "paper/article_manifest.json").read_text())
        if manifest["docx_sha256"] != digest(ROOT / "Portfolio_optimization.docx"):
            raise ValueError("DOCX hash mismatch")
        return {"characters_with_spaces": chars, "abstract_words": words, "references": manifest["references"], "equations": manifest["native_equations"]}

    check("article_formal_and_source_integrity", article)
    report["warnings"].append({"type": "external_snapshot_dependency", "detail": "Exact third-party replication requires the byte-identical 1.3 GB SEC Company Facts archive whose SHA-256 is recorded and locally verified."})
    report["status"] = "failed" if report["failures"] else "passed_with_disclosed_data_dependency"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.bootstrap)
    content = json.dumps(result, indent=2, default=str) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    print(content)
    raise SystemExit(bool(result["failures"]))
