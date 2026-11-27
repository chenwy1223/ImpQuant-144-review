"""Literal frozen native tensor/input/grid validation utilities."""
import hashlib,json,math
from pathlib import Path
from common import require,sha

def tensor_digest(value):
    import torch
    value=value.detach().cpu().contiguous()
    return hashlib.sha256(str((str(value.dtype),tuple(value.shape))).encode()+value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def state_root(records):
    return hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def fingerprint(model):
    return {key:dict(dtype=str(value.dtype),shape=list(value.shape),sha256=tensor_digest(value))
            for key,value in model.state_dict().items()}


def file_record(path):
    p=Path(path);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))


def check_ref(ref):
    path=Path(ref['path'])
    require(path.stat().st_size==ref['bytes'] and sha(path)==ref['sha256'],'Immutable source changed: '+str(path))


def causal_positions(prompt_length,answer_positions,answer_ids,full_ids,labels):
    assert type(prompt_length)is int and prompt_length>0 and answer_ids
    assert all(type(x)is int and x>=0 for x in answer_ids)
    assert answer_positions==list(range(prompt_length,len(full_ids)))
    assert full_ids[prompt_length:]==answer_ids
    assert labels==[-100]*prompt_length+answer_ids
    return [x-1 for x in answer_positions]


def verify_A_equivalence(reference,observed):
    """Use measured bit hashes, not rounded NLL or loose numerical closeness."""
    keys=['id','task','role','answer_positions','causal_prediction_positions',
          'target_token_ids','input_tensor_hashes','labels_sha256',
          'supervised_logits_sha256','direct_loss_fp32_sha256','answer_tokens',
          'target_logits_fp32','logsumexp_fp32','direct_cross_entropy_mean']
    assert {k:reference[k] for k in keys}=={k:observed[k] for k in keys},'Native teacher forward differs from original measured reference'
    return True


def scalar_diagnostics(selected,answer_ids,options,correct_option,eos):
    import torch
    from torch.nn import functional as F
    assert selected.ndim==2 and selected.shape[0]==len(answer_ids) and selected.dtype==torch.float32
    assert bool(torch.isfinite(selected).all()) and all(0<=x<selected.shape[1] for x in answer_ids)
    target=torch.tensor(answer_ids,device=selected.device,dtype=torch.long)
    target_logits=selected.gather(1,target[:,None]).squeeze(1)
    logsumexp=torch.logsumexp(selected,dim=-1)
    direct=F.cross_entropy(selected,target,reduction='mean')
    assert bool(torch.isfinite(direct))
    values=target_logits.detach().cpu().tolist();normalizers=logsumexp.detach().cpu().tolist()
    logprobs=[float(x)-float(z) for x,z in zip(values,normalizers)]
    nll=-sum(logprobs)/len(logprobs)
    assert math.isfinite(nll) and nll>=-2e-5 and math.isclose(nll,float(direct.detach()),rel_tol=2e-5,abs_tol=2e-5)
    first=selected[0];z=normalizers[0];eos_value=float(first[eos].detach())
    detail=dict(eos_token_id=eos,eos_logit_fp32=eos_value,eos_logprob=eos_value-z,
        eos_probability=math.exp(min(0.,eos_value-z)),target_first_token_id=answer_ids[0],
        target_first_token_logprob=logprobs[0],target_first_token_probability=math.exp(min(0.,logprobs[0])),
        full_vocab_argmax_tokenid=int(first.detach().argmax()),full_vocab_logsumexp_fp32=z)
    if options:
        names=list(options)
        assert 2<=len(names)<=5 and names==[chr(65+i) for i in range(len(names))]
        assert correct_option in options and options[correct_option]==answer_ids and len(answer_ids)==1
        assert all(len(ids)==1 for ids in options.values()) and len({ids[0] for ids in options.values()})==len(options)
        scores={name:float(first[ids[0]].detach()) for name,ids in options.items()}
        wrong=max(value for name,value in scores.items() if name!=correct_option);best=max(scores.values())
        best_name=next(name for name,value in scores.items() if value==best)
        x=max(scores.values());conditional=x+math.log(sum(math.exp(value-x) for value in scores.values()))-scores[correct_option]
        detail.update(option_token_ids=options,option_logits_fp32=scores,
            option_logprobs={name:value-z for name,value in scores.items()},correct_option=correct_option,
            strongest_wrong_logit_fp32=wrong,margin_logprob=scores[correct_option]-wrong,
            rank_correct_option=1+sum(value>scores[correct_option] for value in scores.values()),
            best_choice=best_name,option_argmax_correct=int(best_name==correct_option),
            top_option_tie_count=sum(value==best for value in scores.values()),option_conditional_nll=conditional,
            choice_tie_policy='First original A-to-last label wins argmax ties; rank counts strictly greater scores')
    else:
        detail.update(option_token_ids={},option_logits_fp32={},option_logprobs={},correct_option=None,
            strongest_wrong_logit_fp32=None,margin_logprob=None,rank_correct_option=None,best_choice=None,
            option_argmax_correct=None,top_option_tie_count=None,option_conditional_nll=None)
    record=dict(target_logits_fp32=values,logsumexp_fp32=normalizers,target_logprobs_fp32=logprobs,
        answer_nll_token_mean=nll,answer_nll=nll,direct_cross_entropy_mean=float(direct.detach()),
        direct_loss_fp32_sha256=tensor_digest(direct),supervised_logits_sha256=tensor_digest(selected),
        supervised_logits_dtype='torch.float32',answer_tokens=len(answer_ids),first_answer=detail)
    return direct,record


def load_prepared(row):
    import torch
    ref=row['prepared'];p=Path(ref['path'])
    assert p.stat().st_size==ref['bytes'] and sha(p)==ref['sha256']
    sample=torch.load(p,map_location='cpu',weights_only=True)
    assert sample['id']==row['id'] and sample['task']==row['task'] and sample['role']==row['role']
    assert sample['doc_id']==row['doc_id']
    source_keys=['id','task','doc_id','source_index','source_doc_id','role','target','metadata','prompt','formatted_prompt','sha256','pixel_sha256']
    assert sample['source_identity']=={k:row[k] for k in source_keys},'Prepared source question identity changed'
    inputs=sample['inputs'];prefix=sample['prefix_inputs'];labels=sample['labels']
    assert set(inputs)==set(prefix)=={'input_ids','attention_mask','pixel_values','image_grid_thw'}
    assert all(str(v.device)=='cpu' and not v.requires_grad for v in list(inputs.values())+list(prefix.values())+[labels])
    assert labels.dtype==inputs['input_ids'].dtype==torch.long and labels.shape==inputs['input_ids'].shape
    assert labels.shape[0]==1 and inputs['attention_mask'].shape==labels.shape and bool(inputs['attention_mask'].bool().all())
    assert inputs['image_grid_thw'].shape==(1,3)
    for key in ['prompt_length','answer_positions','answer_token_ids','option_token_ids','canonical_target']:
        assert sample[key]==row[key],'Frozen target metadata changed '+key
    pfx=sample['prompt_length'];ids=inputs['input_ids'][0].tolist()
    positions=causal_positions(pfx,sample['answer_positions'],sample['answer_token_ids'],ids,labels[0].tolist())
    assert ids==row['full_input_ids'] and ids[:pfx]==sample['prefix_input_ids']==row['prefix_input_ids']
    assert prefix['input_ids'][0].tolist()==sample['prefix_input_ids']
    measured={k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=tensor_digest(v)) for k,v in dict(inputs,labels=labels).items()}
    prefix_records={k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=tensor_digest(v)) for k,v in prefix.items()}
    assert measured==ref['tensor_records'] and prefix_records==ref['prefix_tensor_records']==row['prefix_tensor_records']
    assert not any(k.endswith('labels') for k in inputs)
    for key in ['pixel_values','image_grid_thw']:
        assert torch.equal(inputs[key].view(torch.uint8),prefix[key].view(torch.uint8))
    return sample,positions,measured,prefix_records


def decode_grid(grid, codes=None):
    """Decode original W3 integer codes with the original frozen FP32 grids."""
    import torch
    q = grid['codes'] if codes is None else codes
    require(q.dtype == torch.uint8 and list(q.shape) == list(grid['shape']),
            'Original integer shape and dtype')
    require(grid['group_size'] == 128 and grid['min_code'] == 0 and
            grid['max_code'] == 7 and grid['weight_dtype'] == 'torch.bfloat16',
            'Original fixed W3 group128 format')
    q = q.detach().cpu()
    scales = grid['scales'].detach().cpu()
    zeros = grid['zeros'].detach().cpu()
    require(scales.dtype == zeros.dtype == torch.float32 and
            scales.shape == zeros.shape == (q.shape[0], q.shape[1] // 128),
            'Original FP32 grid metadata')
    require(q.shape[1] % 128 == 0 and bool(torch.isfinite(scales).all()) and
            bool((scales > 0).all()) and bool(torch.isfinite(zeros).all()) and
            bool((q <= 7).all()), 'Finite legal original grid')
    return (scales.unsqueeze(-1) *
            (q.float().reshape(q.shape[0], -1, 128) - zeros.unsqueeze(-1))).reshape(q.shape).to(torch.bfloat16)


def validate_actual_repaired_weight(grid, value, receipt, expected_count=32):
    """Audit saved integer edits, never choose or generate any new repair."""
    import torch
    original = decode_grid(grid)
    rows, columns = grid['shape']
    require(value.dtype == torch.bfloat16 and tuple(value.shape) == (rows, columns),
            'Fixed saved BF16 weight shape')
    require(tensor_digest(original) == receipt['baseline_weight_sha256'] and
            tensor_digest(value) == receipt['modified_weight_sha256'],
            'Saved actual weight hashes')
    require(receipt['codes_changed'] == receipt['effective_BF16_changes'] == expected_count and
            receipt['original_grid_unchanged'] is True,
            'Exactly requested effective existing code edits per module')
    for key, tensor in [('original_codes_sha256', grid['codes']),
                        ('scales_sha256', grid['scales']), ('zeros_sha256', grid['zeros'])]:
        require(tensor_digest(tensor) == receipt[key], 'Original grid digest: ' + key)
    edits = receipt['edits']
    require(len(edits) == len({(e['row'], e['column']) for e in edits}) == expected_count,
            'Exactly requested distinct existing edit coordinates')
    codes = grid['codes'].detach().cpu().clone()
    for edit in edits:
        row, column = edit['row'], edit['column']
        require(type(row) is int and type(column) is int and
                0 <= row < rows and 0 <= column < columns, 'Saved edit coordinate')
        require(type(edit['direction']) is int and edit['direction'] in (-1, 1) and
                1 <= edit['old_q'] <= 6 and int(codes[row, column]) == edit['old_q'] and
                edit['new_q'] == edit['old_q'] + edit['direction'] and
                0 <= edit['new_q'] <= 7, 'One saved legal neighboring integer code')
        require(float(original[row, column]) == edit['old_value'], 'Exact original BF16 value')
        codes[row, column] = edit['new_q']
    reconstructed = decode_grid(grid, codes)
    require(tensor_digest(codes) == receipt['modified_codes_sha256'], 'Saved modified integer codes')
    require(torch.equal(reconstructed.view(torch.uint8), value.view(torch.uint8)),
            'Actual saved weight must exactly decode from original grid and saved neighboring codes')
    require(int((reconstructed != original).sum()) == expected_count, 'Every requested code move is BF16-effective')
    for edit in edits:
        row, column = edit['row'], edit['column']
        require(float(reconstructed[row, column]) == edit['new_value'] and
                float(reconstructed[row, column]) - float(original[row, column]) == edit['delta_eff'],
                'Saved actual BF16 delta unchanged')
    return original


def raw_state_changes(baseline,active,modules):
    require(set(baseline)==set(active),'Every complete model-state key remains present')
    changed=sorted(k for k in baseline if baseline[k]!=active[k])
    require(changed==sorted(name+'.weight' for name in modules),
            'Only this fixed subset of saved weight tensors may change')
    return changed
