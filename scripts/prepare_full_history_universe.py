"""Create the complete-history research universe from the original OHLCV snapshot.

The eligible universe is defined mechanically: a ticker must have exactly one
finite, positive adjusted-close observation on every date in the source file.
The script also creates a reproducible ticker-to-CIK reference from a dated SEC
exchange-ticker snapshot plus explicit archived-issuer overrides.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OHLCV = (
    Path.home()
    / ".cache/kagglehub/datasets/jacksaleeby/s-and-p500-historical-data/versions/1"
    / "SP500_Historical_Data.csv"
)
SEC_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_SNAPSHOT = ROOT / "reference/company_tickers_exchange_2026-09-07.json"
SOURCE_COLUMNS = ["Ticker", "Date", "Adj Close"]
SPLITS = {
    "train.csv": ("2000-01-03", "2016-12-30"),
    "val.csv": ("2017-01-03", "2020-12-31"),
    "test.csv": ("2021-01-04", "2026-02-20"),
}

# These securities are absent from the current exchange-ticker feed because of
# subsequent ticker changes or listing changes. The CIKs point to the issuer
# whose historical price series is labelled with the dataset ticker.
CIK_OVERRIDES = {
    "AVB": (915912, "AVALONBAY COMMUNITIES INC"),
    "BK": (1390777, "Bank of New York Mellon Corp"),
    "CTRA": (858470, "Coterra Energy Inc."),
    "EQR": (906107, "EQUITY RESIDENTIAL"),
    "HOLX": (859737, "HOLOGIC INC"),
    "SEE": (1012100, "SEALED AIR CORP/DE"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_sec_snapshot(path: Path) -> None:
    request = Request(
        SEC_URL,
        headers={"User-Agent": "PortfolioResearch/1.0 reproducibility research@example.com"},
    )
    with urlopen(request, timeout=60) as response:
        payload = response.read()
    parsed = json.loads(payload)
    if parsed.get("fields") != ["cik", "name", "ticker", "exchange"]:
        raise ValueError("Unexpected SEC ticker snapshot schema")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def load_complete_history(source: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(source, usecols=SOURCE_COLUMNS, parse_dates=["Date"])
    if data.duplicated(["Date", "Ticker"]).any():
        raise ValueError("Duplicate date-ticker keys in the source OHLCV file")
    date_count = data["Date"].nunique()
    profile = data.groupby("Ticker", sort=True).agg(
        rows=("Date", "size"),
        unique_dates=("Date", "nunique"),
        first_date=("Date", "min"),
        last_date=("Date", "max"),
        missing_adjusted_close=("Adj Close", lambda values: int(values.isna().sum())),
        nonpositive_adjusted_close=("Adj Close", lambda values: int((values <= 0).sum())),
        nonfinite_adjusted_close=("Adj Close", lambda values: int((~np.isfinite(values)).sum())),
    )
    eligible = profile.index[
        profile["rows"].eq(date_count)
        & profile["unique_dates"].eq(date_count)
        & profile["missing_adjusted_close"].eq(0)
        & profile["nonpositive_adjusted_close"].eq(0)
        & profile["nonfinite_adjusted_close"].eq(0)
    ]
    selected = data[data["Ticker"].isin(eligible)].sort_values(["Ticker", "Date"])
    expected = len(eligible) * date_count
    if len(selected) != expected:
        raise ValueError(f"Expected {expected:,} selected rows, found {len(selected):,}")
    complete = selected.pivot(index="Date", columns="Ticker", values="Adj Close")
    if complete.shape != (date_count, len(eligible)) or complete.isna().any().any():
        raise ValueError("Selected universe does not form a complete date-ticker panel")
    profile["eligible_complete_history"] = profile.index.isin(eligible)
    return selected, profile.reset_index()


def build_cik_mapping(tickers: list[str], snapshot_path: Path) -> list[dict]:
    snapshot = json.loads(snapshot_path.read_text())
    rows = [dict(zip(snapshot["fields"], row)) for row in snapshot["data"]]
    current = {row["ticker"].replace("-", "."): row for row in rows}
    mapping = []
    for ticker in tickers:
        if ticker in current:
            row = current[ticker]
            cik, name = int(row["cik"]), row["name"]
            source = SEC_URL
        elif ticker in CIK_OVERRIDES:
            cik, name = CIK_OVERRIDES[ticker]
            source = f"https://www.sec.gov/Archives/edgar/data/{cik}/"
        else:
            raise ValueError(f"No SEC CIK mapping for eligible ticker {ticker}")
        company_file = ROOT / "data/company_raw/companyfacts" / f"CIK{cik:010d}.json"
        if not company_file.exists():
            raise FileNotFoundError(company_file)
        company = json.loads(company_file.read_text())
        if int(company["cik"]) != cik:
            raise ValueError(f"CIK mismatch in {company_file.name}")
        mapping.append(
            {
                "ticker": ticker,
                "cik": cik,
                "entity_name": company.get("entityName", name),
                "source": source,
                "identity_only": True,
            }
        )
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ohlcv", type=Path, default=DEFAULT_OHLCV)
    parser.add_argument("--sec-snapshot", type=Path, default=SEC_SNAPSHOT)
    parser.add_argument("--refresh-sec-snapshot", action="store_true")
    args = parser.parse_args()

    if not args.ohlcv.exists():
        raise FileNotFoundError(args.ohlcv)
    if args.refresh_sec_snapshot or not args.sec_snapshot.exists():
        download_sec_snapshot(args.sec_snapshot)

    selected, profile = load_complete_history(args.ohlcv)
    dates = pd.DatetimeIndex(sorted(selected["Date"].unique()))
    if len(dates) != 6573:
        raise ValueError(f"Expected the source maximum of 6,573 dates, found {len(dates):,}")
    tickers = sorted(selected["Ticker"].unique())
    mapping = build_cik_mapping(tickers, args.sec_snapshot)

    for file_name, (start, end) in SPLITS.items():
        part = selected[selected["Date"].between(start, end)]
        expected_dates = dates[(dates >= start) & (dates <= end)]
        if len(part) != len(tickers) * len(expected_dates):
            raise ValueError(f"Incomplete split {file_name}")
        part.to_csv(ROOT / "data" / file_name, index=False)

    (ROOT / "reference/ticker_cik.json").write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False) + "\n"
    )
    table_dir = ROOT / "artifacts/tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    profile.to_csv(table_dir / "universe_source_profile.csv", index=False)
    audit = {
        "source_file": str(args.ohlcv),
        "source_sha256": sha256(args.ohlcv),
        "sec_snapshot": (
            str(args.sec_snapshot.relative_to(ROOT))
            if args.sec_snapshot.is_relative_to(ROOT)
            else str(args.sec_snapshot)
        ),
        "sec_snapshot_sha256": sha256(args.sec_snapshot),
        "source_rows": int(profile["rows"].sum()),
        "source_tickers": int(len(profile)),
        "source_dates": int(len(dates)),
        "selection_rule": "exactly one finite positive adjusted close on every source date",
        "eligible_tickers": int(len(tickers)),
        "eligible_rows": int(len(selected)),
        "first_date": str(dates.min().date()),
        "last_date": str(dates.max().date()),
        "split_rows": {
            file_name: int(pd.read_csv(ROOT / "data" / file_name, usecols=["Date"]).shape[0])
            for file_name in SPLITS
        },
    }
    (table_dir / "universe_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
