"""Stream one archived milestone at a time; verify, evaluate, then release its copies.

Runs on the Mac with gh authentication and SSH access to Unicorn. It never
modifies archived weights or the original training directories. Epoch-50 jobs
and restored copies are retained. Every other checkpoint is restored to the
analysis-only weights/ tree and released only after all eight evaluations finish.
"""
import concurrent.futures,fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path.home()/'ConvRML_Code_Publish/experiments/experiment2_uncertainty_analysis_20261019'
WORK=Path.home()/'.local/share/convrml-exp2-analysis-20261019'
REMOTE='/home/jy2228/ConvRML/experiments/experiment2_uncertainty_analysis_20261019'
SSH=['ssh','-i',str(Path.home()/'.ssh/id_ed25519'),'-o','BatchMode=yes','-o','ConnectTimeout=20','jy2228@unicorn-login-02.coecis.cornell.edu']
METHODS=['im2im','ensemble5','mc_005','qutcc'];CAMERAS=['rml','diffuser'];ARMS=['im2im','mc_005']+[f'ensemble_{i:02d}' for i in range(5)]
def run(cmd,**kw):return subprocess.check_output(cmd,text=True,**kw).strip()
def remote(code):return json.loads(run(SSH+['python3 -'],input=code,timeout=120))
def state_write(d):
 p=WORK/'milestone_state.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def entries(epoch):
 result=[]
 for c in CAMERAS:
  manifest=json.loads((Path.home()/'.local/share/convrml-checkpoint-archive'/f'{c}-manifest.json').read_text())
  for row in manifest['files']:
   p=Path(row['path'])
   if p.name==f'epoch_{epoch:03d}.pt' and p.parts[1] in ARMS:result.append((manifest['release'],row))
 assert len(result)==14
 return result

def restore(item,folder):
 tag,row=item;rel=Path(row['path']);target=folder/rel;target.parent.mkdir(parents=True,exist_ok=True)
 if not target.exists():
  run(['gh','release','download',tag,'--repo','Shdakudow/ConvRML-Uncertainty','--pattern',row['name'],'--dir',str(target.parent),'--clobber'],timeout=3600)
  downloaded=target.parent/row['name'];assert downloaded.stat().st_size==row['size']
  with downloaded.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==row['sha256']
  downloaded.replace(target)
 return target

def main():
 WORK.mkdir(exist_ok=True)
 with (WORK/'milestone.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  file=WORK/'milestone_state.json';state=json.loads(file.read_text()) if file.exists() else {'rounds':{},'epoch50_jobs_reused':True,'test_images':5000,'parallel_streams':8,'primary_smoothing_sigma':0}
  for epoch in range(5,50,5):
   key=str(epoch);row=state['rounds'].get(key,{})
   if row.get('status')=='complete':continue
   if row.get('jobs'):
    probe="import pathlib,json; root=pathlib.Path('"+REMOTE+"/results');out={}\nfor c in ['rml','diffuser']:\n for m in ['im2im','ensemble5','mc_005','qutcc']:\n  p=root/c/m/'epoch_%03d'/'summary.json'\n  if p.exists():\n   r=json.loads(p.read_text());assert r['epoch']==%d and r['n_images']==5000;out[c+'/'+m]=r['test_id_sequence_sha256']\nprint(json.dumps(out))"%(epoch,epoch)
    done=remote(probe)
    if len(done)<8:print('Milestone',epoch,'complete',len(done),'/8',flush=True);return
    assert all(len({v for k,v in done.items() if k.startswith(c+'/')})==1 for c in CAMERAS),'Different test identities within a camera; inspect manifests'
    # Only delete the analysis-owned copies of this finished, non-50 milestone.
    paths=[REMOTE+'/weights/'+r['path'] for _,r in entries(epoch)]
    cleanup='import pathlib,json\npaths='+repr(paths)+"\nfor name in paths:\n p=pathlib.Path(name);assert '/experiment2_uncertainty_analysis_20261019/weights/' in str(p) and p.name=='epoch_%03d.pt';p.unlink(missing_ok=True)\nprint(json.dumps({'cleaned':len(paths)}))"%epoch
    remote(cleanup)
    folder=WORK/'milestones'/f'epoch_{epoch:03d}'
    for _,entry in entries(epoch):(folder/entry['path']).unlink(missing_ok=True)
    row['status']='complete';row['completed_at']=time.time();state['rounds'][key]=row;state_write(state);continue
   folder=WORK/'milestones'/f'epoch_{epoch:03d}';items=entries(epoch)
   # The system Python on this Mac lacks hashlib.file_digest; use the same
   # streaming SHA-256 implementation when that helper is absent.
   with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(lambda item:restore(item,folder),items))
   for c in CAMERAS:
    run(['rsync','-az','-e','ssh -i '+str(Path.home()/'.ssh/id_ed25519')+' -o BatchMode=yes',str(folder/c)+'/',f'jy2228@unicorn-login-02.coecis.cornell.edu:{REMOTE}/weights/{c}/'],timeout=7200)
   expected={r['path']:r['sha256'] for _,r in items}
   submit="""import pathlib,json,subprocess,hashlib
root=pathlib.Path(REMOTE);jobsfile=root/('milestone_jobs_%03d.json'%%EPOCH)
jobs=json.loads(jobsfile.read_text()) if jobsfile.exists() else {}
for name,digest in EXPECTED.items():
 p=root/'weights'/name
 with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest
original=json.loads((root/'jobs.json').read_text())
for c in ['rml','diffuser']:
 for m in ['im2im','ensemble5','mc_005','qutcc']:
  if c+'/'+m in jobs:continue
  old=original[c+'/'+m]
  cmd=['/usr/local/slurm/current/bin/sbatch','--parsable','--qos=normal','--nice=0','--job-name=exp2curve-'+c+'-'+m+'-'+str(EPOCH)]
  if m!='qutcc':cmd+=['--dependency=afterany:'+old]
  cmd+=['run.sbatch',c,m,str(EPOCH)]
  jobs[c+'/'+m]=subprocess.check_output(cmd,cwd=root,text=True).strip().split(';')[0]
  jobsfile.write_text(json.dumps(jobs,indent=2))
print(json.dumps(jobs))
""".replace('%%','%').replace('REMOTE',repr(REMOTE)).replace('EPOCH',str(epoch)).replace('EXPECTED',repr(expected))
   jobs=remote(submit);state['rounds'][key]={'status':'submitted','jobs':jobs,'submitted_at':time.time()};state_write(state);print('Submitted milestone',epoch,jobs,flush=True);return
  state['status']='complete';state_write(state)
if __name__=='__main__':
 if not hasattr(hashlib,'file_digest'):
  def file_digest(stream,algorithm):
   h=hashlib.new(algorithm)
   for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
   return h
  hashlib.file_digest=file_digest
 main()
