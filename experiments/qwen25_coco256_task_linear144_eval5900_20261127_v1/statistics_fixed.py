"""Equal-task-weight paired statistics with joint draws for both contrasts."""
import numpy as np

from common import ARMS,BASELINE,LINEAR,OLD144,BOOTSTRAP_SEED,TASKS,TASK_COUNTS

DRAWS=20_000
CONTRASTS=(
    ("repair_minus_coco256", LINEAR, BASELINE),
    ("repair_minus_old8_repair", LINEAR, OLD144),
)


def ci(x):
    return np.quantile(x,[.025,.975],axis=0).T.tolist()


def summarize(rows,mapped):
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    image_rng=np.random.default_rng(BOOTSTRAP_SEED+1)
    by_task={}
    mean=None
    grouped_mean=None
    for task in TASKS:
        data=[r for r in rows if r["task"]==task]
        n=TASK_COUNTS[task]
        assert n==len(data)
        scores={a:np.array([mapped[a][r["id"]]["score"] for r in data],dtype=np.float64) for a in ARMS}
        delta=np.column_stack([scores[hi]-scores[lo] for _,hi,lo in CONTRASTS])
        _,group=np.unique([r["visual_group"] for r in data],return_inverse=True)
        counts=np.bincount(group)
        sums=np.column_stack([np.bincount(group,weights=delta[:,j]) for j in range(len(CONTRASTS))])
        p=np.empty((DRAWS,len(CONTRASTS)))
        g=np.empty_like(p)
        for start in range(0,DRAWS,500):
            end=min(start+500,DRAWS)
            ind=rng.integers(n,size=(end-start,n))
            p[start:end]=delta[ind].mean(axis=1)*100
            ind=image_rng.integers(len(counts),size=(end-start,len(counts)))
            g[start:end]=sums[ind].sum(axis=1)*100/counts[ind].sum(axis=1)[:,None]
        paired_ci,group_ci=ci(p),ci(g)
        metrics={}
        for j,(key,_,_) in enumerate(CONTRASTS):
            plus=int((delta[:,j]>1e-12).sum())
            minus=int((delta[:,j]<-1e-12).sum())
            metrics[key]=dict(delta_pp=float(delta[:,j].mean()*100),improved=plus,declined=minus,unchanged=n-plus-minus,
                              paired_CI95_pp=paired_ci[j],image_group_CI95_pp=group_ci[j])
        by_task[task]=dict(n=n,arms={a:dict(pct=float(s.mean()*100),full=int((s==1).sum()),
                                          partial=int(((s>0)&(s<1)).sum()),zero=int((s==0).sum()))
                                    for a,s in scores.items()},
                           contrasts=metrics,distinct_image_sets=len(counts))
        by_task[task]["generation"]={}
        for arm in ARMS:
            records=[mapped[arm][r["id"]] for r in data]
            by_task[task]["generation"][arm]=dict(
                first_token_eos=sum(r["generated_token_ids"][0]==r["generation_kwargs"]["eos_token_id"] for r in records),
                empty=sum(not r["response"].strip() for r in records))

        if mean is None:
            mean,grouped_mean=p,g
        else:
            mean+=p
            grouped_mean+=g
    macro={a:float(np.mean([t["arms"][a]["pct"] for t in by_task.values()])) for a in ARMS}
    paired_ci=ci(mean/len(TASKS))
    group_ci=ci(grouped_mean/len(TASKS))
    return dict(status="COMPLETE_THREE_ARM_PAIRED_5900",
                tasks=by_task,macro=macro,macro_contrasts={
                    key:dict(delta_pp=macro[hi]-macro[lo],paired_CI95_pp=paired_ci[j],image_group_CI95_pp=group_ci[j])
                    for j,(key,hi,lo) in enumerate(CONTRASTS)
                },bootstrap=dict(draws=DRAWS,seed=BOOTSTRAP_SEED,image_group_seed=BOOTSTRAP_SEED+1,
                                 common_within_task_indices=True))
