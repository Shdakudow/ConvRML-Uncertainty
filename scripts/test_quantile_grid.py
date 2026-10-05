import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from models.convnext import FNN,QuantileConditioner
from models.quantile_grid import GridHead,fresh_head,pinball,LEVELS
torch.set_num_threads(2);torch.manual_seed(7)
head=fresh_head(FNN(16,3,(16,16,8,8)))
grid=GridHead(head);conditioner=QuantileConditioner(16,8,4)
f=torch.randn(2,16,8,8);x=torch.rand(2,3,8,8)
p=grid(f)
reference=torch.stack([head(conditioner(f,torch.full((2,),q))) for q in LEVELS],1)
torch.testing.assert_close(p,reference)
assert p.shape==(2,5,3,8,8)
pinball(p,x).backward()
assert grid.head.f[-1].weight.grad.abs().sum()>0
# Sequential backward and the joint five-level loss must have equal gradients.
head.zero_grad();conditioner.zero_grad();pinball(reference,x).backward()
expected=head.f[-1].weight.grad.clone();head.zero_grad();conditioner.zero_grad()
for q in LEVELS:
    (pinball(head(conditioner(f,torch.full((2,),q)))[:,None],x,(q,))/5).backward()
torch.testing.assert_close(head.f[-1].weight.grad,expected)
print('PASS: matched predictions, shape, gradients, five-level accumulation')
