"""Run real notebook cells in fresh kernels, with socket-free and normal Jupyter backends."""
from pathlib import Path
import os,sys,json,time,argparse,subprocess
ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'artifacts/cache';CACHE.mkdir(parents=True,exist_ok=True)
for key,sub in [('JUPYTER_CONFIG_DIR','jupyter-config'),('JUPYTER_RUNTIME_DIR','jupyter-runtime'),('IPYTHONDIR','ipython'),('MPLCONFIGDIR','matplotlib')]:
 p=CACHE/sub;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['PYDEVD_DISABLE_FILE_VALIDATION']='1'
import nbformat
parser=argparse.ArgumentParser();parser.add_argument('--backend',choices=['inprocess','jupyter'],default='inprocess');parser.add_argument('--single');args=parser.parse_args()
names=['baseline.ipynb','pipeline1_regime_aware_gmv.ipynb','pipeline2_fundamental_quality_gmv.ipynb','pipeline3_catboost_risk_gmv.ipynb','results_comparison.ipynb']

def _execute_one(name):
 os.chdir(ROOT);sys.path.insert(0,str(ROOT));path=ROOT/name;nb=nbformat.read(path,as_version=4);t=time.time()
 for cell in nb.cells:
  if cell.cell_type=='code':cell.outputs=[];cell.execution_count=None
 print('EXECUTING',name,flush=True)
 if args.backend=='jupyter':
  from nbclient import NotebookClient
  client=NotebookClient(nb,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}},allow_errors=False)
  try:client.execute()
  finally:nbformat.write(nb,path)
 else:
  from ipykernel.inprocess.manager import InProcessKernelManager
  km=InProcessKernelManager();km.start_kernel();kc=km.client();kc.start_channels()
  def drain(msg_id,cell=None):
   error=None
   while True:
    m=kc.get_iopub_msg(timeout=1800)
    if m['parent_header'].get('msg_id')!=msg_id:continue
    kind=m['msg_type'];content=m['content']
    if kind=='execute_input' and cell is not None:cell.execution_count=content['execution_count']
    if kind in ('stream','display_data','execute_result','error') and cell is not None:
     cell.outputs.append(nbformat.v4.output_from_msg(m))
    if kind=='error':error='\n'.join(content.get('traceback',[]))
    if kind=='status' and content['execution_state']=='idle':break
   if error:raise RuntimeError(error)
  try:
   drain(kc.execute('%matplotlib inline',silent=True,store_history=False))
   for number,cell in enumerate(nb.cells):
    if cell.cell_type!='code':continue
    print(f'  cell {number+1}/{len(nb.cells)}',flush=True)
    drain(kc.execute(cell.source,store_history=True),cell)
    nbformat.write(nb,path)
  finally:
   nbformat.write(nb,path);kc.stop_channels();km.shutdown_kernel()
 errors=[o for c in nb.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error']
 if errors or not all(c.execution_count is not None for c in nb.cells if c.cell_type=='code'):
  raise RuntimeError('Notebook contains failed or unexecuted code cells')
 nbformat.validate(nb)
 record={'notebook':name,'status':'completed','backend':args.backend,'code_cells':sum(c.cell_type=='code' for c in nb.cells),'seconds':round(time.time()-t,2)}
 (CACHE/(name+'.execution.json')).write_text(json.dumps(record))
 print('COMPLETED',record,flush=True)

def update_execution_status(status,name,error=None):
 p=ROOT/'artifacts/run_manifest.json'
 manifest=json.loads(p.read_text()) if p.exists() else {}
 # Preserve the previous experiment metadata while invalidating its execution claim.
 manifest['status']=status
 manifest['latest_notebook_attempt']={'notebook':name,'status':status,'backend':args.backend}
 if error:manifest['latest_notebook_attempt']['error']=error
 p.write_text(json.dumps(manifest,indent=2)+'\n')

def execute_one(name):
 path=(ROOT/name).resolve()
 if path.parent!=ROOT or path.suffix!='.ipynb' or not path.is_file():
  raise ValueError('Notebook must be an existing .ipynb in the repository root')
 name=path.name
 record_path=CACHE/(path.name+'.execution.json')
 record_path.write_text(json.dumps({'notebook':name,'status':'running','backend':args.backend}))
 update_execution_status('running',name)
 try:
  _execute_one(name)
 except BaseException as exc:
  error=f'{type(exc).__name__}: {exc}'
  record_path.write_text(json.dumps({'notebook':name,'status':'failed','backend':args.backend,'error':error}))
  update_execution_status('failed',name,error)
  raise
 # One notebook cannot certify that the entire experiment has been rerun.
 update_execution_status('notebooks_partial',name)

if args.single:
 execute_one(args.single)
else:
 for name in names:
  subprocess.run([sys.executable,__file__,'--backend',args.backend,'--single',name],check=True)
 records=[json.loads((CACHE/(name+'.execution.json')).read_text()) for name in names]
 p=ROOT/'artifacts/run_manifest.json';manifest=json.loads(p.read_text());manifest['notebook_execution']=records;manifest['status']='completed'
 manifest['latest_notebook_attempt']={'status':'completed','backend':args.backend,'notebooks':names}
 p.write_text(json.dumps(manifest,indent=2)+'\n')
 print('ALL NOTEBOOKS EXECUTED SUCCESSFULLY',flush=True)
