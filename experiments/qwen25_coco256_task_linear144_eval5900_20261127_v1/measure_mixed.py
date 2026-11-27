"""Original QA native-prefix route plus native answer-only OCR CE."""
import time
from common import *
from native_contracts import tensor_digest
from native_decision import measure as qa_measure,serialize_first_kwargs,prefix_forward,gradient_one as qa_gradient
from decision_math import processed_scores

def ocr_teacher(model,row,prefix):
 import torch
 import torch.nn.functional as F
 from inputs_native import teacher_inputs
 full,positions=teacher_inputs(prefix,row['answer_token_ids']);device=model.get_submodule(TARGET_MODULE).weight.device
 gpu={k:v.to(device) for k,v in full.items()}
 output=model(**gpu,labels=None,use_cache=False,past_key_values=None,output_attentions=False,output_hidden_states=False,return_dict=True)
 assert output.loss is None and output.past_key_values is None
 selected=output.logits[0,positions,:].float();labels=torch.tensor(row['answer_token_ids'],device=device,dtype=torch.long)
 loss=F.cross_entropy(selected,labels,reduction='mean');target=selected.gather(1,labels[:,None]).squeeze(1);lse=torch.logsumexp(selected,dim=1)
 info=dict(answer_nll=float(loss.detach()),target_logits_fp32=target.detach().cpu().tolist(),logsumexp_fp32=lse.detach().cpu().tolist(),supervised_logits_sha256=tensor_digest(selected),answer_tokens=len(row['answer_token_ids']),causal_prediction_positions=positions,teacher_input_tensor_records={k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=tensor_digest(v)) for k,v in full.items()},teacher_rule='Mean CE over literal canonical answer tokens only; no prompt/image/pad/EOS supervision')
 return loss,info

def measure(model,adapter,row,prefix,arm,role,root,manifest_sha,save_full=False):
 import torch
 from supervision_metrics import score_response,generation_flags
 if row['answer_kind']=='choice':
  r=qa_measure(model,adapter,row,prefix,arm,role,root,manifest_sha,save_full=save_full);r['score']=float(r['correct']);return r
 device=model.get_submodule(TARGET_MODULE).weight.device;gpu={k:v.to(device) for k,v in prefix.items()};captured=[]
 def hook(module,args,kwargs):
  if not captured:captured.append(serialize_first_kwargs(args,kwargs,prefix))
 handle=model.register_forward_pre_hook(hook,with_kwargs=True);kwargs=row['generation_kwargs'];tick=time.monotonic()
 assert kwargs==dict(max_new_tokens=128,do_sample=False,num_beams=1,use_cache=True,eos_token_id=adapter.tokenizer.eos_token_id,pad_token_id=adapter.tokenizer.pad_token_id)
 assert model.generation_config.repetition_penalty==1.05
 try:
  with torch.no_grad():output=model.generate(**gpu,**kwargs,return_dict_in_generate=True,output_scores=True,output_logits=True)
 finally:handle.remove()
 assert len(captured)==1;sequence=output.sequences;n=prefix['input_ids'].shape[1];assert torch.equal(sequence[0,:n],gpu['input_ids'][0])
 ids=sequence[0,n:].detach().cpu().tolist();raw=output.logits[0][0].detach().float().clone();scores=output.scores[0][0].detach().float().clone();replay=processed_scores(raw,gpu['input_ids'])
 assert torch.equal(replay.view(torch.uint8),scores.view(torch.uint8)) and int(scores.argmax())==ids[0]
 with torch.no_grad():checked=prefix_forward(model,prefix,captured[0],device);loss,teacher=ocr_teacher(model,row,prefix)
 assert torch.equal(raw.view(torch.uint8),checked.view(torch.uint8))
 text=adapter.processor.batch_decode([ids],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0];response=text.split(adapter.tokenizer.decode(kwargs['eos_token_id']))[0]
 aliases=row['metadata']['aliases'];score=score_response(row['task'],response,aliases,row['metadata']);flags=generation_flags(row['task'],response,ids,kwargs,score['score'],[])
 full=None
 if save_full:
  p=OUT/'S_firststep_vectors'/('%03d.pt'%row['split_index']);p.parent.mkdir(exist_ok=True);assert not p.exists();torch.save(dict(id=row['id'],raw_logits_FP32=raw.cpu(),processed_scores_FP32=scores.cpu(),native_forward_extras=captured[0]),p);full=ref(p)
 value=dict(attempt=NAME,utc=utc(),arm=arm,role=role,id=row['id'],task=row['task'],source_index=row['source_index'],split=row['role'],image_sha256=row['sha256'],pixel_sha256=row['pixel_sha256'],target=row['target'],aliases=aliases,manifest_sha256=manifest_sha,model_state_root=root,generation_kwargs=kwargs,native_forward_extras=captured[0],prefix_tensor_records=row['prefix_tensor_records'],raw_native_prefix_bit_equal=True,actual_processed_scores_bit_equal=True,repetition_penalty=1.05,raw_logits_sha256=tensor_digest(raw),processed_scores_sha256=tensor_digest(scores),independent_prefix_logits_sha256=tensor_digest(checked),replayed_processed_scores_sha256=tensor_digest(replay),processed_argmax_token=int(scores.argmax()),generated_token_ids=ids,decoded_response=text,response=response,score=score['score'],flags=flags,firststep_vector_file=full,seconds=time.monotonic()-tick,**teacher)
 del output,sequence,raw,scores,replay,checked,gpu,loss
 return value

def gradient_one(model,row,prefix,reference,route,module):
 import torch
 if row['answer_kind']=='choice':return qa_gradient(model,row,prefix,reference,'nll32',route,module)
 before=tensor_digest(model.get_submodule(module).weight)
 with route._temporary_route(model,module) as (handle,checkpoint,restoration):
  with torch.enable_grad():
   loss,r=ocr_teacher(model,row,prefix)
   assert r=={k:reference[k] for k in r},'OCR native gradient forward differs from independent answer-only teacher'
   assert loss.requires_grad
   grad=torch.autograd.grad(loss,handle.target_weight,create_graph=False,retain_graph=False,allow_unused=False)[0]
  assert grad.dtype==torch.bfloat16 and tuple(grad.shape)==(512,3584) and bool(torch.isfinite(grad).all())
  result=grad.detach().cpu().contiguous();assert all(v>0 for v in checkpoint.report()['checkpoint_forward_calls'].values()) and handle.target_weight.grad is None
  del grad,loss
 assert tensor_digest(model.get_submodule(module).weight)==before
 return result,dict(objective='OCR_answer_only_token_mean_CE',id=row['id'],native_raw_bits_equal=True,native_processed_bits_equal=True,native_teacher_logits_bit_equal=True,objective_value=reference['answer_nll'],supervised_logits_sha256=reference['supervised_logits_sha256'],gradient_sha256=tensor_digest(result),route_restoration=restoration)
