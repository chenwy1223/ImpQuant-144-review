"""Native CPU input identities, fixed literal answer-only teachers, no tensor caches."""
import copy
from common import *
from native_contracts import tensor_digest
_PROCESSOR=None

def processor():
 global _PROCESSOR
 if _PROCESSOR is None:
  from transformers import AutoProcessor
  _PROCESSOR=AutoProcessor.from_pretrained(str(MODEL),local_files_only=True,min_pixels=200704,max_pixels=1003520)
 return _PROCESSOR

def build(row,proc=None):
 import torch
 from processor_compat import canonical_inputs
 proc=processor() if proc is None else proc
 assert sha(row['image'])==row['sha256'] and row['task'] in TASKS and row['role'] in SPLIT_COUNTS
 chat,prefix,compat=canonical_inputs(proc,row);prefix={k:v.cpu().contiguous() for k,v in prefix.items()}
 assert set(prefix)=={'input_ids','attention_mask','pixel_values','image_grid_thw'} and bool((prefix['attention_mask']==1).all())
 ids=proc.tokenizer.encode(row['canonical_target'],add_special_tokens=False);assert ids and not set(ids)&set(proc.tokenizer.all_special_ids)
 assert proc.tokenizer.encode(chat+row['canonical_target'],add_special_tokens=False)==proc.tokenizer.encode(chat,add_special_tokens=False)+ids
 opts={k:proc.tokenizer.encode(k,add_special_tokens=False) for k in row['metadata'].get('options',{})} if row['answer_kind']=='choice' else {}
 if opts:assert len(ids)==1 and all(len(v)==1 for v in opts.values()) and opts[row['target']]==ids and len({v[0] for v in opts.values()})==len(opts)
 records={k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=tensor_digest(v)) for k,v in prefix.items()}
 return prefix,dict(chat_prompt=chat,prompt_length=prefix['input_ids'].shape[1],prefix_tensor_records=records,prefix_input_ids=prefix['input_ids'][0].tolist(),answer_token_ids=ids,option_token_ids=opts,processor_compatibility=compat,generation_kwargs=dict(max_new_tokens=16 if opts else 128,do_sample=False,num_beams=1,use_cache=True,eos_token_id=proc.tokenizer.eos_token_id,pad_token_id=proc.tokenizer.pad_token_id))

def seal(old,proc=None):
 row=copy.deepcopy(old);row['image']=str(HERE/'source_data_v1'/row['relative_image']);_,identity=build(row,proc);row.update(identity);return row

def load(row):
 prefix,identity=build(row);assert identity=={k:row[k] for k in identity},'Native input differs from CPU seal';return prefix

def teacher_inputs(prefix,ids):
 import torch
 n=prefix['input_ids'].shape[1];full=dict(prefix);full['input_ids']=torch.cat([prefix['input_ids'],torch.tensor([ids],dtype=torch.long)],dim=1);full['attention_mask']=torch.ones_like(full['input_ids']);return full,list(range(n-1,n+len(ids)-1))
