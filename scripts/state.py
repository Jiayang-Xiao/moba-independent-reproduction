import json
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
from common import save_json
p=Path(sys.argv[1]);old=json.loads(p.read_text()) if p.exists() else {}
old.update({'status':sys.argv[2]})
if len(sys.argv)>3:old['last_exit_code']=int(sys.argv[3])
save_json(p,old)
