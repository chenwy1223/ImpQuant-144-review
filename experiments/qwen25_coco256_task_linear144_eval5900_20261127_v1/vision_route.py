"""Exact vision attention with a bounded query workspace for large documents."""
import ast
from pathlib import Path

from common import OUT, save, sha
from eval_inputs import prefixes, processor
from native_contracts import tensor_digest

NATIVE_PATCH_LIMIT = 5120
QUERY_BLOCK = 128
EXPECTED_UPSTREAM_SHA = "076a9b98955abe003ad90f9240372996bbc72d9f24e303346f23b77735106cd7"
DISPATCH = {"tiled_calls":0,"max_patches":0}


def tiled(q,k,v,cu_seqlens):
    import torch
    from torch.nn import functional as F

    n=q.shape[1]
    boundaries=cu_seqlens.detach().cpu().tolist()
    assert boundaries[0]==0 and boundaries[-1]==n
    segments=list(zip(boundaries,boundaries[1:]))
    rows=[]
    for start in range(0,n,QUERY_BLOCK):
        end=min(start+QUERY_BLOCK,n)
        mask=torch.zeros((1,end-start,n),device=q.device,dtype=torch.bool)
        for left,right in segments:
            a,b=max(start,left),min(end,right)
            if a<b:
                mask[...,a-start:b-start,left:right]=True
        queries=q[:,start:end]
        length=end-start
        if length<QUERY_BLOCK and n>QUERY_BLOCK:
            extra=QUERY_BLOCK-length
            queries=torch.cat((queries,queries[:,-1:].expand(-1,extra,-1)),dim=1)
            mask=torch.cat((mask,mask[:,-1:].expand(-1,extra,-1)),dim=1)
        rows.append(F.scaled_dot_product_attention(queries,k,v,mask,dropout_p=0.0)[:,:length])
    return torch.cat(rows,dim=1)


def install_and_verify(model, rows, phase):
    import torch
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as mod

    source=Path(mod.__file__)
    assert sha(source)==EXPECTED_UPSTREAM_SHA
    tree=ast.parse(source.read_text(encoding="utf-8"))
    clazz="Qwen2_5_VLVisionSdpaAttention"
    class_ast=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==clazz)
    fn=next(x for x in class_ast.body if isinstance(x,ast.FunctionDef) and x.name=="forward")
    cls=getattr(mod,clazz)
    assert sum(type(m) is cls for m in model.visual.modules())>0
    original=cls.forward
    body=[]
    removed=replaced=0
    for stmt in fn.body:
        if isinstance(stmt,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="attention_mask" for t in stmt.targets):
            removed+=1
            continue
        if isinstance(stmt,ast.For) and "attention_mask" in ast.unparse(stmt):
            removed+=1
            continue
        if isinstance(stmt,ast.Assign) and isinstance(stmt.value,ast.Call) and isinstance(stmt.value.func,ast.Attribute) and stmt.value.func.attr=="scaled_dot_product_attention":
            stmt.value=ast.parse("TILED_SDPA(q,k,v,cu_seqlens)",mode="eval").body
            replaced+=1
        body.append(stmt)
    assert removed==2 and replaced==1
    fn.body=body
    ast_module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    env=dict(mod.__dict__,TILED_SDPA=tiled)
    exec(compile(ast_module,"<memory-bounded native vision SDPA>","exec"),env)
    forward_tiled=env["forward"]

    def dispatch(self, hidden_states, *args, **kwargs):
        n=hidden_states.shape[0]
        if n<=NATIVE_PATCH_LIMIT:
            return original(self,hidden_states,*args,**kwargs)
        DISPATCH["tiled_calls"]+=1
        DISPATCH["max_patches"]=max(DISPATCH["max_patches"],n)
        return forward_tiled(self,hidden_states,*args,**kwargs)

    device=next(model.visual.parameters()).device
    dtype=model.visual.dtype
    evidence=[]
    ids=("V:mmmu_val:51","V:mmmu_val:83")
    for rid in ids:
        row=next(r for r in rows if r["id"]==rid)
        phase("native_tiled_real_vision_match",task="mmmu_val",sample_id=rid)
        inp=prefixes(row,processor())
        pix=inp["pixel_values"].to(device=device,dtype=dtype)
        grid=inp["image_grid_thw"].to(device)
        n=pix.shape[0]
        assert n<=NATIVE_PATCH_LIMIT and (n>4096 or len(grid)>1)
        with torch.no_grad():
            reference=model.visual(pix,grid_thw=grid).detach().cpu()
            try:
                cls.forward=forward_tiled
                candidate=model.visual(pix,grid_thw=grid).detach().cpu()
            finally:
                cls.forward=original
        assert reference.dtype==candidate.dtype==dtype
        assert torch.equal(reference.view(torch.uint8),candidate.view(torch.uint8)),rid
        assert torch.cuda.max_memory_allocated()/2**30<=22.5
        evidence.append(dict(id=rid,images=len(grid),raw_patches=int(n),shape=list(reference.shape),
                             bytewise_equal=True,sha256=tensor_digest(reference)))
        del pix,grid
        torch.cuda.empty_cache()
    cls.forward=dispatch
    save(OUT/"vision_equivalence.json",dict(status="PASS",cases=evidence,native_limit=NATIVE_PATCH_LIMIT,
         query_block=QUERY_BLOCK,upstream_sha256=EXPECTED_UPSTREAM_SHA,full_keys=True,block_mask=True))
