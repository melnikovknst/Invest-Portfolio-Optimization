"""Acquire or verify the exact raw inputs used by the published experiment."""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRICE_SHA256 = "82f3ab3ab6821c54b67a555c1f52c60dc4c162923499f2a1403c51d08944722a"
SEC_ZIP_SHA256 = "a62a7c6a9bfea0d721a73ac9439b8a07a089da1eeee01aa71ef06917e8be77eb"
SEC_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def locate_price(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if os.environ.get("PORTFOLIO_OHLCV"):
        candidates.append(Path(os.environ["PORTFOLIO_OHLCV"]).expanduser())
    candidates.extend([
        ROOT / "data/SP500_Historical_Data.csv",
        Path.home() / ".cache/kagglehub/datasets/jacksaleeby/s-and-p500-historical-data/versions/1/SP500_Historical_Data.csv",
    ])
    for path in candidates:
        if path.is_file():
            return path.resolve()
    try:
        import kagglehub
        folder = Path(kagglehub.dataset_download("jacksaleeby/s-and-p500-historical-data"))
        path = folder / "SP500_Historical_Data.csv"
        if path.is_file():
            return path.resolve()
    except Exception as exc:
        raise FileNotFoundError("Price snapshot unavailable; pass --price or configure Kaggle credentials.") from exc
    raise FileNotFoundError("SP500_Historical_Data.csv was not found in the downloaded Kaggle dataset.")


def verify(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: expected {expected}, found {actual} at {path}")
    print(f"verified {label}: {actual}  {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--price", help="Path to the Kaggle version-1 OHLCV CSV")
    parser.add_argument("--companyfacts-zip", help="Path to the byte-identical SEC archive")
    parser.add_argument("--download-current-sec", action="store_true", help="Download the current SEC archive; strict hash verification still applies")
    parser.add_argument("--extract", action="store_true", help="Extract the verified SEC archive into data/company_raw/companyfacts")
    args = parser.parse_args()

    price = locate_price(args.price)
    verify(price, PRICE_SHA256, "price snapshot")

    destination = ROOT / "data/company_raw/companyfacts.zip"
    supplied = Path(args.companyfacts_zip).expanduser().resolve() if args.companyfacts_zip else destination
    if not supplied.is_file() and args.download_current_sec:
        user_agent = os.environ.get("SEC_USER_AGENT")
        if not user_agent:
            raise ValueError(
                "Set SEC_USER_AGENT to a descriptive application name and monitored contact address "
                "before downloading from sec.gov."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(SEC_URL, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(request) as response, destination.open("wb") as out:
            shutil.copyfileobj(response, out)
        supplied = destination
    if not supplied.is_file():
        raise FileNotFoundError("Provide --companyfacts-zip with the archived SEC snapshot or use the local verified archive.")
    verify(supplied, SEC_ZIP_SHA256, "SEC Company Facts archive")

    if args.extract:
        target = ROOT / "data/company_raw/companyfacts"
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(supplied) as archive:
            archive.extractall(target)
        print(f"extracted {len(list(target.glob('CIK*.json'))):,} issuer files to {target}")


if __name__ == "__main__":
    main()
