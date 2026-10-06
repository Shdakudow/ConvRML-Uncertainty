"""Prepare two isolated epoch-20 controls, preserving original AdamW moments."""
import sys,json,copy,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from models.shamus_rml import ShamusRML
from models.formal_quantile import QuantileRML
torch.set_num_threads(2)
source=Path('results/formal_quantile_comparison')
root=Path('results/controls30_20260924')
if (root/'prepared.json').exists():
    print('Already prepared');raise SystemExit
ck=torch.load(source/'normal/latest.pt',map_location='cpu',weights_only=False)
assert ck['epoch']==20
low=root/'mse_lr1e5';(low/'normal').mkdir(parents=True,exist_ok=True)
for file in ['latest.pt','best.pt','history.json']:shutil.copy2(source/'normal'/file,low/'normal'/file)
shutil.copy2(source/'smoke_normal.json',low/'smoke_normal.json')
joint=root/'joint_w002';(joint/'conditioned').mkdir(parents=True,exist_ok=True)
torch.manual_seed(20260924)
base=ShamusRML();base.load_state_dict(ck['model']);model=QuantileRML(base,'conditioned')
opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
state=opt.state_dict();old=ck['optimizer'];oldids=old['param_groups'][0]['params'];newids=state['param_groups'][0]['params']
assert len(oldids)==len(list(base.parameters()))
assert [name for name,_ in model.named_parameters()][:len(oldids)]==['base.'+name for name,_ in base.named_parameters()]
state['state']={new:copy.deepcopy(old['state'][prev]) for prev,new in zip(oldids,newids) if prev in old['state']}
state['param_groups'][0]={**old['param_groups'][0],'params':newids}
opt.load_state_dict(state)
# Keep all source moments exactly; new conditioner/head have no moments yet.
for prev,new in zip(oldids,newids):
    for key,val in old['state'].get(prev,{}).items():
        if torch.is_tensor(val):assert torch.equal(val,state['state'][new][key])
current_mse=ck['history'][-1]['validation_loss']
ck.update(model=model.state_dict(),optimizer=opt.state_dict(),best=current_mse)
torch.save(ck,joint/'conditioned/latest.pt')
torch.save({'model':model.state_dict(),'epoch':20,'validation':current_mse,'trained_stage':'conditioned','manifest_sha256':ck['manifest_sha256']},joint/'conditioned/best.pt')
(joint/'conditioned/history.json').write_text(json.dumps(ck['history'],indent=2))
shutil.copy2(source/'smoke_conditioned.json',joint/'smoke_conditioned.json')
record={'source':str(source/'normal/latest.pt'),'source_epoch':20,'target_epoch':30,
        'mse_lr1e5':{'kind':'normal','lr':1e-5},
        'joint_w002':{'kind':'conditioned','lr':1e-4,'pinball_weight':.02,'q_sampling':'uniform','selection':'validation MSE','base_optimizer_moments':'copied exactly','new_head_optimizer_moments':'fresh'},
        'comparison':'exact epoch 30 versus exact epoch 20','manifest_sha256':ck['manifest_sha256']}
(root/'prepared.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
for directory in [low,joint]:(directory/'prepared.json').write_text(json.dumps(record,indent=2))
