"""Capture actual first-generation-call kwargs, then verify native prefix logits."""
import math
from common import *
from native_contracts import tensor_digest
from decision_math import common_statistics,processed_scores,objective

def serialize_first_kwargs(args,kwargs,prefix):
    import torch
    assert not args and all(k in kwargs for k in prefix)
    extras={}
    for k,v in kwargs.items():
        if k in prefix:
            assert torch.equal(v.detach().cpu().contiguous().view(torch.uint8),prefix[k].view(torch.uint8)),('Actual native prefix changed',k)
        elif k=='past_key_values':
            assert v is None or v.get_seq_length()==0,'Only empty native initial cache'
            extras[k]=None
        elif k=='use_cache':assert v is True;extras[k]=False
        elif isinstance(v,torch.Tensor):
            assert v.numel()<=100000 and v.dtype==torch.int64,('Unexpected native positional tensor',k)
            extras[k]=dict(tensor_values=v.detach().cpu().tolist(),dtype='torch.int64')
        else:
            assert v is None or type(v) in (str,int,float,bool),('Unsupported actual native call argument',k)
            extras[k]=v
    assert extras['use_cache'] is False
    extras['past_key_values']=None
    return extras

def forward_kwargs(prefix,extras,device):
    import torch
    result={k:v.to(device) for k,v in prefix.items()}
    for k,v in extras.items():
        result[k]=torch.tensor(v['tensor_values'],device=device,dtype=torch.long) if isinstance(v,dict) else v
    assert result['use_cache'] is False and result['past_key_values'] is None
    return result

def prefix_forward(model,prefix,extras,device):
    kw=forward_kwargs(prefix,extras,device)
    out=model(**kw)
    assert out.loss is None and out.past_key_values is None
    return out.logits[0,-1,:].float()

def measure(model,adapter,row,prefix,arm,role,root,manifest_sha,*,save_full=False):
    import torch,time
    from supervision_metrics import score_response,generation_flags
    device=model.get_submodule(TARGET_MODULE).weight.device
    gpu={k:v.to(device) for k,v in prefix.items()};captured=[]
    def hook(module,args,kwargs):
        if not captured:captured.append(serialize_first_kwargs(args,kwargs,prefix))
    handle=model.register_forward_pre_hook(hook,with_kwargs=True)
    kwargs=row['generation_kwargs'];tick=time.monotonic()
    assert kwargs==dict(max_new_tokens=16,do_sample=False,num_beams=1,use_cache=True,eos_token_id=adapter.tokenizer.eos_token_id,pad_token_id=adapter.tokenizer.pad_token_id)
    assert model.generation_config.repetition_penalty==1.05
    try:
        with torch.no_grad():
            output=model.generate(**gpu,**kwargs,return_dict_in_generate=True,output_scores=True,output_logits=True)
    finally:handle.remove()
    assert len(captured)==1
    n=prefix['input_ids'].shape[1];sequence=output.sequences
    assert torch.equal(sequence[0,:n],gpu['input_ids'][0])
    ids=sequence[0,n:].detach().cpu().tolist();raw=output.logits[0][0].detach().float().clone();actual=output.scores[0][0].detach().float().clone()
    replay=processed_scores(raw,gpu['input_ids']);assert torch.equal(replay.view(torch.uint8),actual.view(torch.uint8)),'Actual native logits processor not reproduced'
    assert int(actual.argmax())==ids[0],'Native processed argmax differs from actual greedy token'
    with torch.no_grad():checked=prefix_forward(model,prefix,captured[0],device)
    assert torch.equal(raw.view(torch.uint8),checked.view(torch.uint8)),'Native no-cache prefix forward not bit-identical to real generation first step'
    target=row['answer_token_ids'][0];record=common_statistics(raw,actual,target,row['option_token_ids'],kwargs['eos_token_id'])
    text=adapter.processor.batch_decode([ids],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
    response=text.split(adapter.tokenizer.decode(kwargs['eos_token_id']))[0]
    score=score_response('scienceqa_img2017',response,row['target'],row['metadata'])
    flags=generation_flags('scienceqa_img2017',response,ids,kwargs,score['correct'],list(row['option_token_ids']))
    full=None
    if save_full:
        p=OUT/'S_firststep_vectors'/('%03d.pt'%row['split_index']);p.parent.mkdir(exist_ok=True)
        assert not p.exists()
        torch.save(dict(id=row['id'],raw_logits_FP32=raw.cpu(),processed_scores_FP32=actual.cpu(),native_forward_extras=captured[0]),p)
        full=ref(p)
    value=dict(attempt=NAME,utc=utc(),arm=arm,role=role,id=row['id'],task=row['task'],source_index=row['source_index'],split=row['role'],
      image_sha256=row['sha256'],pixel_sha256=row['pixel_sha256'],target=row['target'],manifest_sha256=manifest_sha,model_state_root=root,
      generation_kwargs=kwargs,native_forward_extras=captured[0],prefix_tensor_records=row['prefix_tensor_records'],
      raw_native_prefix_bit_equal=True,actual_processed_scores_bit_equal=True,repetition_penalty=1.05,
      independent_prefix_logits_sha256=tensor_digest(checked),replayed_processed_scores_sha256=tensor_digest(replay),
      generated_token_ids=ids,decoded_response=text,response=response,correct=score['correct'],flags=flags,
      firststep_vector_file=full,seconds=time.monotonic()-tick,**record)
    del output,sequence,raw,actual,replay,checked,gpu
    return value

def gradient_one(model,row,prefix,reference,arm,route,module):
    import torch
    device=model.get_submodule(module).weight.device
    prefix_gpu={k:v.to(device) for k,v in prefix.items()}
    before=tensor_digest(model.get_submodule(module).weight)
    with route._temporary_route(model,module) as (handle,checkpoint,restoration):
        with torch.enable_grad():
            raw=prefix_forward(model,prefix,reference['native_forward_extras'],device)
            assert tensor_digest(raw)==reference['raw_logits_sha256'],'Gradient-route first native logits changed'
            scores=processed_scores(raw,prefix_gpu['input_ids'])
            assert tensor_digest(scores)==reference['processed_scores_sha256'],'Gradient-route native processed scores changed'
            loss,rival=objective(raw,prefix_gpu['input_ids'],row['answer_token_ids'][0],arm)
            expected=reference['answer_nll'] if arm=='nll32' else reference['decision_loss']
            assert float(loss.detach())==expected and loss.requires_grad
            grad=torch.autograd.grad(loss,handle.target_weight,create_graph=False,retain_graph=False)[0]
        assert grad.dtype==torch.bfloat16 and tuple(grad.shape)==(512,3584) and bool(torch.isfinite(grad).all())
        result=grad.detach().cpu().contiguous()
        assert all(v>0 for v in checkpoint.report()['checkpoint_forward_calls'].values())
        assert handle.target_weight.grad is None
        del grad,raw,scores,loss
    assert tensor_digest(model.get_submodule(module).weight)==before
    return result,dict(objective=arm,id=row['id'],native_raw_bits_equal=True,native_processed_bits_equal=True,
      raw_logits_sha256=reference['raw_logits_sha256'],processed_scores_sha256=reference['processed_scores_sha256'],
      strongest_competing_token=rival,objective_value=expected,gradient_sha256=tensor_digest(result),route_restoration=restoration)
