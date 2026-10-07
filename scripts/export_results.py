"""Recompute AUROC before whitelisted publication; write inference weights for Release."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import zipfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import digest, load_json, save_json, sha
from scoring import auroc, video_anomaly_scores


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();out=Path(args.out)
    config=load_json(ROOT/'config.json');summary=load_json(out/'summary.json');fit=load_json(out/'trained.json')
    if summary['status']!='COMPLETED' or summary['identity']!=config['identity'] or fit['checkpoint_sha256']!=sha(out/'last.pt'):
        raise RuntimeError('Cannot publish an incomplete or changed result')
    recomputed={}
    for row in summary['targets']:
        target=row['target'];path=out/(target+'.npz')
        if sha(path)!=row['scores_sha256']:raise RuntimeError('Changed raw target scores')
        with np.load(path,allow_pickle=False) as raw:
            value=auroc(raw['label'],raw['score'])
            if len(raw['label'])!=config['targets'][target]['scored_frames']:raise RuntimeError('Target alignment mismatch')
            for video in np.unique(raw['video']):
                mask=raw['video']==video
                expected=video_anomaly_scores(raw['psnr'][mask],config['score']['smoothing_sigma'])
                if not np.allclose(expected,raw['score'][mask],atol=1e-12,rtol=0):raise RuntimeError('Changed normalization/smoothing')
        if abs(value-row['AUROC'])>1e-12:raise RuntimeError('Raw AUROC differs from summary')
        recomputed[target]=value
    if set(recomputed)!=set(config['targets']) or abs(np.mean(list(recomputed.values()))-summary['macro_AUROC'])>1e-12:
        raise RuntimeError('Wrong macro target set')
    destination=ROOT/'evidence/run';destination.mkdir(parents=True,exist_ok=True)
    existing=destination/'manifest.json'
    model_file=ROOT/'.private/publication/model.pt'
    bundle=ROOT/'.private/publication/review_bundle.zip'
    if existing.exists() and (destination/'summary.json').exists() and sha(destination/'summary.json')==sha(out/'summary.json'):
        record=load_json(existing)
        for name,declared in record['files'].items():
            if sha(destination/name)!=declared['sha256']:raise RuntimeError('Changed published evidence')
        model_record=load_json(destination/'model_release.json')
        if not model_file.exists() or sha(model_file)!=model_record['sha256'] or not bundle.exists():
            raise RuntimeError('Missing/changed publication assets; restore them before retrying')
        print('Identical verified publication evidence reused',flush=True);return
    names=['config.json','preflight.json','data_inventory.json','source_cache_record.json','trained.json',
           'evaluation_freeze.json','results.csv','summary.json','train.jsonl','local_checks_server.json']
    for target in config['targets']:names.extend([target+'.npz',target+'.json'])
    for name in names:shutil.copyfile(out/name,destination/name)
    if (out/'execution.log').exists():
        with (out/'execution.log').open(encoding='utf-8',errors='replace') as f:
            from collections import deque
            tail=deque(f,maxlen=250)
        (destination/'execution_tail.txt').write_text(''.join(tail),encoding='utf-8')
    save_json(destination/'environment.json',{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,
        'cuda':torch.version.cuda,'platform':platform.platform(),
        'packages':sorted({d.metadata['Name']+'=='+d.version for d in importlib.metadata.distributions() if d.metadata['Name']})})
    saved=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
    if saved['context']!=fit['context'] or saved['epoch']!=config['epochs']:raise RuntimeError('Checkpoint context/epoch mismatch')
    private=ROOT/'.private/publication';private.mkdir(parents=True,exist_ok=True)
    model_file=private/'model.pt'
    torch.save({'identity':config['identity'],'config':config,'model':saved['model'],'trained':fit,
                'author_confirmed':False},model_file)
    save_json(destination/'model_release.json',{'filename':'model.pt','sha256':sha(model_file),'bytes':model_file.stat().st_size,
        'format':'torch.load(weights_only=True); model state dict and primitive metadata',
        'publication':'GitHub Release asset; full optimizer checkpoint remains on server'})
    manifest={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in destination.iterdir()
              if p.is_file() and p.name!='manifest.json'}
    save_json(destination/'manifest.json',{'identity':config['identity'],'files':manifest,'recomputed_AUROC':recomputed})
    bundle=private/'review_bundle.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(destination.iterdir()):
            if file.is_file():archive.write(file,'evidence/run/'+file.name)
    print('AUROC independently recomputed and publication evidence exported:',recomputed,flush=True)


if __name__=='__main__':main()
