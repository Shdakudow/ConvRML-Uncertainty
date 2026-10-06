import json
from pathlib import Path
source=Path('results/learning_curve_sources')
rows=[]
for camera in ('rml','diffuser'):
 for f in sorted((source/camera).glob('epoch_*/*.json')):
  d=json.loads(f.read_text())
  assert d['test_count']==5000 and d['alpha']==.1 and d['calibration_count']==2048
  rows.append(dict(camera=camera,epoch=d['backbone_epoch'],method=d['method'],calibration='empirical',metrics=d['metrics'],test_count=d['test_count']))
for f in sorted((source/'rcps').glob('*/*/comparison.json')):
 d=json.loads(f.read_text());m=d['metrics']
 rows.append(dict(camera=d['camera'],epoch=d['backbone_epoch'],method=d['method'],calibration='rcps',test_count=d['test_count'],metrics=dict(psnr=m['psnr'],ssim=m['ssim'],mae=m['mae'],coverage=m['rcps_hb_coverage'],width=m['rcps_hb_width'])))
for camera in ('rml','diffuser'):
 for method in ('fixed','conditioned','ensemble','mc'):
  epochs=[r['epoch'] for r in rows if r['camera']==camera and r['method']==method and r['calibration']=='empirical']
  assert epochs==list(range(5,51,5)),(camera,method,epochs)
meta=dict(rows=rows,alpha=.1,rcps_delta=.05,test_count=5000,calibration_count=2048,description='Mean test metrics on all 5,000 test images at saved backbone milestones. Fixed heads: ten extra training epochs at each frozen backbone. Empirical and RCPS series are separate; missing RCPS epochs were not evaluated. Epochs are not equal GPU compute budgets across methods.')
Path('/Users/shdakudow/ConvRML_Demo_Publish/learning_curves.js').write_text('window.LEARNING_CURVES='+json.dumps(meta,separators=(',',':'))+';')
print('Verified and exported',len(rows),'method/checkpoint/calibration records')
