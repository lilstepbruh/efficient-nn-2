"""Pinned FineWeb-100BT -> GPT-2 uint16 streams, content-hash document split."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(ROOT/'data/fineweb/cache'))
os.environ.setdefault('TIKTOKEN_CACHE_DIR',str(ROOT/'data/fineweb/tokenizer_cache'))
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
import argparse,hashlib,json,time
import numpy as np
import tiktoken
from datasets import load_dataset

REVISION='9bb295ddab0e05d785b879661af7260fed5140fc'
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--train-tokens',type=int,default=220_000_000);p.add_argument('--eval-tokens',type=int,default=2_000_000);a=p.parse_args()
 out=ROOT/'data/fineweb';out.mkdir(parents=True,exist_ok=True)
 if (out/'manifest.json').exists():raise RuntimeError('Prepared dataset exists; preserve it or choose a new dataset directory')
 enc=tiktoken.get_encoding('gpt2')
 ds=load_dataset('HuggingFaceFW/fineweb',name='sample-100BT',revision=REVISION,split='train',streaming=True)
 ds=ds.shuffle(seed=20261010,buffer_size=10000)
 caps={'train':a.train_tokens,'validation':a.eval_tokens,'test':a.eval_tokens}
 counts={k:0 for k in caps};docs={k:0 for k in caps};seen=set();duplicates=0;start=time.time()
 handles={k:open(out/f'{k}.bin.partial','wb') for k in caps}
 index=open(out/'documents.jsonl.partial','w')
 try:
  for record in ds:
   text=record['text']; key=hashlib.sha256(text.encode('utf-8')).hexdigest()
   if key in seen:duplicates+=1;continue
   seen.add(key)
   bucket=int(key[:16],16)%1000
   split='validation' if bucket<10 else 'test' if bucket<20 else 'train'
   if counts[split]>=caps[split]:continue
   tokens=enc.encode_ordinary(text)+[enc.eot_token]
   # Keep complete documents; budget may be exceeded by one document.
   np.asarray(tokens,dtype='<u2').tofile(handles[split])
   index.write(json.dumps({'sha256':key,'split':split,'offset':counts[split],'tokens':len(tokens),'id':record.get('id'),'dump':record.get('dump')})+'\n')
   counts[split]+=len(tokens);docs[split]+=1
   if sum(docs.values())%1000==0:print(json.dumps({'tokens':counts,'documents':docs,'elapsed_seconds':round(time.time()-start)}),flush=True)
   if all(counts[k]>=caps[k] for k in counts):break
 finally:
  for f in handles.values():f.close()
  index.close()
 if not all(counts[k]>=caps[k] for k in counts):raise RuntimeError('Dataset exhausted before budgets reached')
 for name in caps:(out/f'{name}.bin.partial').replace(out/f'{name}.bin')
 (out/'documents.jsonl.partial').replace(out/'documents.jsonl')
 manifest={'dataset':'HuggingFaceFW/fineweb','subset':'sample-100BT','revision':REVISION,'shuffle_seed':20261010,'shuffle_buffer':10000,'split':'SHA256 of full document UTF-8 content modulo 1000; validation 0..9, test 10..19, train 20..999','tokenizer':'tiktoken gpt2','tiktoken_version':tiktoken.__version__,'vocab_size':enc.n_vocab,'eot_token':enc.eot_token,'storage_dtype':'little-endian uint16','tokens':counts,'documents':docs,'exact_duplicates_skipped':duplicates,'seconds':time.time()-start,'sha256':{k:digest(out/f'{k}.bin') for k in caps}}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__':main()
