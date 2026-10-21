import json,os,subprocess,sys
from pathlib import Path
t=json.loads(Path('reports/all_rcps_tasks.json').read_text())[int(os.environ['SLURM_ARRAY_TASK_ID'])]
subprocess.run([sys.executable,'scripts/compare_rcps.py','--worker',str(t['worker']),'--epoch',str(t['epoch'])],check=True)
