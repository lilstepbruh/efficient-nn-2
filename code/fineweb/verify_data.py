"""Validate prepared files and enforce document-level split separation."""
import hashlib,json
from pathlib import Path
from .train import ROOT

def verify():
 folder=ROOT/'data/fineweb';m=json.loads((folder/'manifest.json').read_text())
 if m['dataset']!='HuggingFaceFW/fineweb' or m['subset']!='sample-100BT' or m['tokenizer']!='tiktoken gpt2':raise ValueError('Unexpected dataset/tokenizer')
 counts={k:0 for k in ['train','validation','test']};documents={k:0 for k in counts};seen=set()
 for line in (folder/'documents.jsonl').open():
  d=json.loads(line);key=d['sha256'];split=d['split'];bucket=int(key[:16],16)%1000
  expected='validation' if bucket<10 else 'test' if bucket<20 else 'train'
  if split!=expected or key in seen or d['offset']!=counts[split]:raise ValueError('Duplicate, split contamination or invalid document offset')
  seen.add(key);counts[split]+=d['tokens'];documents[split]+=1
 for split,n in counts.items():
  path=folder/f'{split}.bin'
  if path.stat().st_size!=n*2 or n!=m['tokens'][split] or documents[split]!=m['documents'][split]:raise ValueError('Token/document size mismatch')
  h=hashlib.sha256()
  with path.open('rb') as f:
   for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
  if h.hexdigest()!=m['sha256'][split]:raise ValueError('Checksum mismatch: '+split)
 print(json.dumps({'verified_data':counts,'exact_duplicate_documents':0}),flush=True)
 return m
if __name__=='__main__':verify()
