"""Optional explicit copy to a clean checkout; never stages or removes raw data."""
from pathlib import Path
import argparse,json,shutil,subprocess
SOURCE=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--destination',type=Path,required=True)
TARGET=parser.parse_args().destination.resolve()
FILES=['.gitignore','README.md','requirements.txt','requirements-lock.txt','baseline.ipynb','pipeline1_regime_aware_gmv.ipynb','pipeline2_fundamental_quality_gmv.ipynb','pipeline3_catboost_risk_gmv.ipynb','results_comparison.ipynb']
FILES+=['Portfolio optimization.docx','REVIEW.md','requirements-docs.txt']
DIRS=['portfolio_research','reference','scripts','tests','paper']
if SOURCE.resolve()==TARGET.resolve():raise SystemExit('Already in the original repository.')
if not (TARGET/'.git').exists():raise SystemExit('Expected original repository not found.')
state=subprocess.run(['git','status','--porcelain','--untracked-files=all'],cwd=TARGET,capture_output=True,text=True,check=True)
if state.stdout:raise SystemExit('Refusing to overwrite a destination with modified or untracked files; review its changes first.')
manifest=json.loads((SOURCE/'artifacts/run_manifest.json').read_text())
if manifest['status']!='completed':raise SystemExit('The reviewed run is not complete.')
for name in FILES:shutil.copy2(SOURCE/name,TARGET/name)
for name in DIRS:
 shutil.copytree(SOURCE/name,TARGET/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','rendered'))
for name in ['results','tables','figures','review']:
 if (SOURCE/'artifacts'/name).exists():shutil.copytree(SOURCE/'artifacts'/name,TARGET/'artifacts'/name,dirs_exist_ok=True)
for name in ['frozen_spec.json','run_manifest.json']:
 shutil.copy2(SOURCE/'artifacts'/name,TARGET/'artifacts'/name)
print(f'Copied reviewed notebooks, code and results to {TARGET}. Raw data and source EDA notebooks were preserved. No commit or push was made.')
