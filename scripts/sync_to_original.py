"""Copy the reviewed deliverables to the user's original repository, preserving all raw data.

Run this script directly only if the managed workspace cannot write to Desktop/work.
It refuses to overwrite a tracked source file changed since the original checkout.
"""
from pathlib import Path
import hashlib,json,shutil,subprocess
SOURCE=Path(__file__).resolve().parents[1]
TARGET=Path('/Users/konstantinmelnikov/Desktop/work/Invest-Portfolio-Optimization')
FILES=['.gitignore','README.md','requirements.txt','requirements-lock.txt','baseline.ipynb','pipeline1_regime_aware_gmv.ipynb','pipeline2_fundamental_quality_gmv.ipynb','pipeline3_catboost_risk_gmv.ipynb','results_comparison.ipynb']
DIRS=['portfolio_research','reference','scripts','tests']
if SOURCE.resolve()==TARGET.resolve():raise SystemExit('Already in the original repository.')
if not (TARGET/'.git').exists():raise SystemExit('Expected original repository not found.')
for name in ['baseline.ipynb','pipeline1_regime_aware_gmv.ipynb','requirements.txt']:
 changed=subprocess.run(['git','diff','--quiet','HEAD','--',name],cwd=TARGET).returncode
 if changed and (TARGET/name).read_bytes()!=(SOURCE/name).read_bytes():raise SystemExit(f'Refusing to overwrite your modified {name}; review the working-copy diff first.')
manifest=json.loads((SOURCE/'artifacts/run_manifest.json').read_text())
if manifest['status']!='completed':raise SystemExit('The reviewed run is not complete.')
for name in FILES:shutil.copy2(SOURCE/name,TARGET/name)
for name in DIRS:
 shutil.copytree(SOURCE/name,TARGET/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ['results','tables','figures']:
 shutil.copytree(SOURCE/'artifacts'/name,TARGET/'artifacts'/name,dirs_exist_ok=True)
for name in ['frozen_spec.json','run_manifest.json']:
 shutil.copy2(SOURCE/'artifacts'/name,TARGET/'artifacts'/name)
environment=SOURCE.parent/'.venv'
if environment.exists() and not (TARGET/'.venv').exists():
 (TARGET/'.venv').symlink_to(environment,target_is_directory=True)
# --cached changes tracking only: the three local CSVs and the entire SEC archive remain intact.
subprocess.run(['git','rm','--cached','--ignore-unmatch','--','data/train.csv','data/val.csv','data/test.csv'],cwd=TARGET,check=True)
print(f'Copied reviewed notebooks, code and results to {TARGET}. Raw data and source EDA notebooks were preserved. No commit or push was made.')
