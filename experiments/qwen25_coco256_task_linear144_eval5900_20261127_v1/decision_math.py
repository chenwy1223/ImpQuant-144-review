"""Actual greedy first-step objective, including the original repetition penalty."""
def processed_scores(logits,prefix_ids,penalty=1.05):
    import torch
    assert logits.ndim==1 and logits.dtype==torch.float32 and bool(torch.isfinite(logits).all())
    assert penalty==1.05 and prefix_ids.dtype==torch.long
    # One transform per vocabulary entry: repeated visual/prompt IDs must not
    # duplicate backward contributions through a scatter with repeated indices.
    seen=torch.zeros(logits.numel(),dtype=torch.bool,device=logits.device)
    seen[prefix_ids.flatten().unique()]=True
    adjusted=torch.where(logits<0,logits*penalty,logits/penalty)
    return torch.where(seen,adjusted,logits)

def objective(logits,prefix_ids,target,arm):
    import torch
    from torch.nn import functional as F
    assert arm in ('nll32','decision32') and 0<=target<logits.numel()
    if arm=='nll32':return F.cross_entropy(logits[None,:],torch.tensor([target],device=logits.device)),None
    scores=processed_scores(logits,prefix_ids)
    wrong=scores.detach().clone();wrong[target]=float('-inf')
    rival=int(wrong.argmax())
    return F.softplus(scores[rival]-scores[target]),rival

def common_statistics(raw,scores,target,options,eos):
    import math
    import torch
    from torch.nn import functional as F
    from native_contracts import tensor_digest
    assert raw.shape==scores.shape and raw.dtype==scores.dtype==torch.float32
    rival_scores=scores.detach().clone();rival_scores[target]=float('-inf');rival=int(rival_scores.argmax())
    z=float(torch.logsumexp(raw,0));nll=float(F.cross_entropy(raw[None,:],torch.tensor([target],device=raw.device)))
    option_logits={k:float(raw[v[0]]) for k,v in options.items()}
    correct=next(k for k,v in options.items() if v==[target]);wrong=max(v for k,v in option_logits.items() if k!=correct)
    return dict(answer_nll=nll,raw_logsumexp=z,target_raw_logit=float(raw[target]),
      target_processed_score=float(scores[target]),strongest_competing_token=rival,strongest_competing_score=float(scores[rival]),
      decision_loss=float(F.softplus(scores[rival]-scores[target])),processed_margin=float(scores[target]-scores[rival]),
      EOS_raw_logit=float(raw[eos]),EOS_processed_score=float(scores[eos]),EOS_probability=math.exp(float(raw[eos])-z),
      raw_argmax_token=int(raw.argmax()),processed_argmax_token=int(scores.argmax()),
      raw_logits_sha256=tensor_digest(raw),processed_scores_sha256=tensor_digest(scores),
      option_logits=option_logits,option_margin=option_logits[correct]-wrong,correct_option=correct,
      target_token_id=target,eos_token_id=eos)
