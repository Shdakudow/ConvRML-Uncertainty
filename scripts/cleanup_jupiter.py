"""Delete only authorized Jupiter experiment artifacts; never shared data."""
import json,shutil
from pathlib import Path
root=Path(__file__).resolve().parents[1]
paths=[root/'results/Jupiters_exp',root/'checkpoints/Jupiters_exp',root/'models/jupiter_shamus.py',root/'reports/Jupiters_exp_protocol.md',root/'configs/PLD_RML/config_Jupiters_exp.yaml']
for folder in ('scripts','logs'):
    paths.extend(p for p in (root/folder).iterdir() if 'jupiter' in p.name.lower() and p.name!='cleanup_jupiter.py')
paths.extend(p for p in root.iterdir() if p.is_file() and 'jupiter' in p.name.lower())
accidental=root/'jy2228@unicorn-login-01.coecis.cornell.edu:'
if accidental.exists():paths.extend(p for p in accidental.rglob('*') if p.is_file() and 'jupiter' in p.name.lower())
report=Path.home()/'Downloads/ConvRML_Toy_Report/reports/Jupiters_exp_protocol.md'
paths.append(report)
removed=[]
for p in paths:
    if not p.exists() and not p.is_symlink():continue
    if p.is_dir() and not p.is_symlink():shutil.rmtree(p)
    else:p.unlink()
    removed.append(str(p))
out=root/'reports';out.mkdir(exist_ok=True)
(out/'cleanup_receipt.json').write_text(json.dumps({'removed':removed,'preserved':'Shared data, Shamus source checkpoint, non-Jupiter experiments'},indent=2))
print(json.dumps(removed,indent=2))
