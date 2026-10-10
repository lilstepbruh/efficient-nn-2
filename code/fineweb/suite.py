"""Run a fixed paired experiment; persist status and stop within the user time budget."""
import argparse,json,os,subprocess,sys,time,fcntl,hashlib
from pathlib import Path
from .train import ROOT,atomic_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--hours',type=float,default=7.5);p.add_argument('--tokens',type=int,default=150_994_944);p.add_argument('--tune-tokens',type=int,default=25_165_824);a=p.parse_args()
 out=ROOT/'results/fineweb_v2';out.mkdir(parents=True,exist_ok=True)
 lock=open(out/'suite.lock','w')
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 started=time.time();deadline=started+a.hours*3600
 plan={'created_unix':started,'wall_budget_hours':a.hours,'main_tokens_per_run':a.tokens,'tune_tokens_per_run':a.tune_tokens,'lr_candidates':[.0003,.0006,.001],'tuning_seed':3407,'seeds':[17,42,123],'contexts':[512,2048],'global_tokens':16384,'micro_tokens':4096,'selection':'minimum validation loss of SA at context 2048; no test access during tuning','main_runs':12,'test':'512 fixed windows, same last 256 targets for all contexts','model':'8 layers, residual width 512, 4 heads x 128, GPT-2 vocabulary; 50,936,320 parameters','projection':'native bf16 operations under autocast, matching Algorithm 1 arithmetic dtype'}
 if (out/'plan.json').exists():
  old=json.loads((out/'plan.json').read_text())
  for key in ['main_tokens_per_run','tune_tokens_per_run','lr_candidates','seeds','contexts']:
   if old[key]!=plan[key]:raise ValueError('Existing plan differs; do not combine experiments')
 else:atomic_json(out/'plan.json',plan)
 status={'state':'running','started_unix':started,'deadline_unix':deadline,'completed':[]}
 def run(name,arguments):
  folder=out/name
  if (folder/'result.json').exists():
   saved=json.loads((folder/'result.json').read_text())
   current_hash={f:hashlib.sha256((ROOT/'fineweb'/f).read_bytes()).hexdigest() for f in ['model.py','train.py']}
   current_data=json.loads((ROOT/'data/fineweb/manifest.json').read_text())
   if saved['meta']['source_sha256']!=current_hash or saved['meta']['data_manifest']['sha256']!=current_data['sha256']:
    raise ValueError('Completed result belongs to different code or data; keep series separate')
   status['completed'].append(name);return
  folder.mkdir(parents=True,exist_ok=True)
  cmd=[sys.executable,'-m','fineweb.train','--out',str(folder.relative_to(ROOT)),*arguments]
  if (folder/'last.pt').exists():cmd.append('--resume')
  if time.time()>=deadline:raise TimeoutError('Wall-time budget exhausted')
  status.update({'active':name,'active_started_unix':time.time(),'command':cmd});atomic_json(out/'status.json',status)
  print(json.dumps({'starting':name,'remaining_minutes':round((deadline-time.time())/60)}),flush=True)
  with open(folder/'run.log','a') as log:
   proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   try:
    while proc.poll() is None:
     if time.time()>deadline:raise TimeoutError('Wall-time budget exhausted')
     status['heartbeat_unix']=time.time();atomic_json(out/'status.json',status);time.sleep(10)
    if proc.returncode:raise RuntimeError(f'{name} failed: inspect {folder}/run.log')
   except BaseException:
    proc.terminate()
    try:proc.wait(timeout=20)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()
    raise
  status['completed'].append(name);atomic_json(out/'status.json',status)
  result=json.loads((folder/'result.json').read_text())
  print(json.dumps({'completed':name,'validation_loss':result['validation']['loss'],'test_loss':result['test']['loss'] if result['test'] else None,'train_seconds':result['train_seconds']}),flush=True)
 try:
  for lr in plan['lr_candidates']:
   run(f'tune_sa_lr{lr:g}',['--mode','SA','--seed','3407','--lr',str(lr),'--tokens',str(a.tune_tokens),'--eval-every','512','--tuning'])
  scores={lr:json.loads((out/f'tune_sa_lr{lr:g}'/'result.json').read_text())['validation']['loss'] for lr in plan['lr_candidates']}
  selected=min(scores,key=scores.get);atomic_json(out/'lr_selection.json',{'validation_scores':scores,'selected_lr':selected,'rule':plan['selection']})
  print(json.dumps({'selected_lr':selected,'scores':scores}),flush=True)
  # Pair adjacent methods, alternate method and context order across seeds.
  for i,seed in enumerate(plan['seeds']):
   for context in ([2048,512] if i%2==0 else [512,2048]):
    for mode in (['SA','XSA'] if i%2==0 else ['XSA','SA']):
     run(f'c{context}_s{seed}_{mode.lower()}',['--mode',mode,'--seed',str(seed),'--context',str(context),'--lr',str(selected),'--tokens',str(a.tokens)])
  status.update({'state':'training_complete','finished_unix':time.time(),'active':None});atomic_json(out/'status.json',status)
 except BaseException as error:
  status.update({'state':'budget_exhausted' if isinstance(error,TimeoutError) else 'failed','error':str(error),'finished_unix':time.time()});atomic_json(out/'status.json',status)
  raise
 finally:
  if (ROOT/'fineweb/summarize.py').exists():subprocess.run([sys.executable,'-m','fineweb.summarize'],cwd=ROOT,check=False)

if __name__=='__main__':main()
