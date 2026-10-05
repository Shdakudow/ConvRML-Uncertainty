"""Plot verified own-experiment widths; do not fill missing dataset results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path.home()/'Downloads'
quantiles=json.loads((root/'ConvRML_Im2im_vs_QConditioned/comparison.json').read_text())
baselines=json.loads((root/'ConvRML_Calibration95/comparison.json').read_text())
rows=[]
for key,label in [('conditioned','Q-conditioned ConvRML'),('fixed','Normal ConvRML + fixed head')]:
    m=quantiles['metrics']['test'][key]['intervals']['calibrated95']
    rows.append({'method':label,'width':m['width'],'coverage':m['coverage']})
for key,label in [('ensemble','Deep ensemble (5 members)'),('mc','MC dropout (30 samples)')]:
    m=baselines['metrics']['test'][key]['calibrated95']
    rows.append({'method':label,'width':m['mean_width'],'coverage':m['coverage']})
out=root/'ConvRML_Interval_Width_Comparison';out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(12,6.8));fig.subplots_adjust(left=.29,right=.96,top=.78,bottom=.28)
y=np.arange(4)
ax.barh(y,[r['width'] for r in rows],height=.55,color=['#197c80','#5898b0','#8293b9','#c7a274'])
ax.set_yticks(y,[r['method'] for r in rows]);ax.invert_yaxis();ax.set_xlim(0,.075)
ax.set_xlabel('Mean calibrated interval width (RGB intensity scale 0–1)')
ax.set_axisbelow(True);ax.grid(axis='x',alpha=.18);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=10)
for i,r in enumerate(rows):
    ax.text(r['width']+.001,i,f"{r['width']:.5f}  |  {r['coverage']:.2%} coverage",va='center',fontsize=10)
fig.text(.06,.93,'Interval widths in your real-RML experiment',fontsize=20,weight='bold')
fig.text(.06,.865,'95% calibration target • same 2,500 test images • 2,048 separate calibration images',fontsize=11,color='#465365')
fig.text(.06,.16,'Lower width is preferable at comparable coverage. These methods use different training procedures and budgets.',fontsize=10,color='#465365')
fig.text(.06,.115,'Poisson noise, Gaussian noise, MRI and QPI: own-experiment results not yet located; no values estimated or substituted.',fontsize=10,color='#465365')
fig.text(.06,.07,'Empirical calibration; average pixel/channel coverage, not a per-image or distribution-free guarantee.',fontsize=10,color='#465365')
for ext in ('png','pdf','svg'):fig.savefig(out/f'interval_width_real_rml.{ext}',dpi=220,facecolor='white')
plt.close(fig)
(out/'verified_values.json').write_text(json.dumps({'dataset':'real RML measurements','target_coverage':.95,'test_count':2500,'rows':rows,'missing_datasets':['Poisson noise','Gaussian noise','MRI','QPI'],'sources':[str(root/'ConvRML_Im2im_vs_QConditioned/comparison.json'),str(root/'ConvRML_Calibration95/comparison.json')]},indent=2))
print(out/'interval_width_real_rml.png')
