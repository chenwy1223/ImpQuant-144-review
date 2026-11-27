import os

from qmllm.methods.awq.entry import awq_entry
from qmllm.methods.smoothquant.entry import smoothquant_entry
from qmllm.methods.mbq.entry import mbq_entry
from qmllm.methods.rtn.entry import rtn_entry
from qmllm.methods.impquant.entry import impquant_entry
from qmllm.methods.gptq.entry import gptq_entry

def qwrapper(model, prompt_inputs, prompt_kwargs, args):
    if args.method == "awq":
        model = awq_entry(model, prompt_inputs, prompt_kwargs, run_awq_process=args.run_process, scale_path=args.scale_path, q_group_size=args.w_group, w_bit=args.w_bit)
    elif args.method == "smoothquant":
        model = smoothquant_entry(
            model,
            prompt_inputs,
            prompt_kwargs,
            run_sq_process=args.run_process,
            pseudo_quant=args.pseudo_quant,
            scale_path=args.scale_path,
            w_bit=args.w_bit,
            a_bit=args.a_bit,
            alpha=args.alpha,
            q_group_size=args.w_group,
        )
    elif args.method == "mbq":
        wa_quant = args.w_bit < 16 and args.a_bit < 16
        model = mbq_entry(model, prompt_inputs, prompt_kwargs,
                                run_mbq_process=args.run_process,
                                pseudo_quant=args.pseudo_quant,
                                scale_path=args.scale_path,
                                q_group_size=args.w_group,
                                w_bit=args.w_bit,
                                a_bit=args.a_bit,
                                wa_quant=wa_quant,
                                reweight=True,
                                loss_mode="mae")
    elif args.method == "impquant":
        model = impquant_entry(
            model,
            prompt_inputs,
            prompt_kwargs,
            run_impquant_process=args.run_process,
            pseudo_quant=args.pseudo_quant,
            scale_path=args.scale_path,
            q_group_size=args.w_group,
            w_bit=args.w_bit,
            a_bit=args.a_bit,
            text_scale=args.impquant_text_scale,
            vision_scale=args.impquant_vision_scale,
            alpha=args.impquant_alpha,
            jsd_eps=args.impquant_jsd_eps,
            outlier_ratio=args.impquant_outlier_ratio,
            percdamp=args.gptq_percdamp,
            probe_cache_path=args.impquant_probe_cache_path,
            probe_source_path=args.impquant_probe_source_path,
            probe_only=args.impquant_probe_only,
            skip_probe1=args.impquant_skip_probe1,
            probe2_mode=args.impquant_probe2_mode,
            probe_group_h=args.impquant_probe_group_h,
            probe_group_w=args.impquant_probe_group_w,
            probe_refine_ratio=args.impquant_probe_refine_ratio,
            probe_refine_strategy=args.impquant_probe_refine_strategy,
            probe_refine_seed=args.impquant_probe_refine_seed,
        )
    elif args.method == "gptq":
        model = gptq_entry(
            model,
            prompt_inputs,
            prompt_kwargs,
            run_gptq_process=args.run_process,
            pseudo_quant=args.pseudo_quant,
            scale_path=args.scale_path,
            blocksize=getattr(args, "gptq_blocksize", 128),
            percdamp=getattr(args, "gptq_percdamp", 0.01),
            groupsize=getattr(args, "gptq_groupsize", -1),
            actorder=getattr(args, "gptq_actorder", False),
            w_bit=args.w_bit,
        )
    elif args.method == "rtn":
        wa_quant = args.w_bit < 16 and args.a_bit < 16
        model = rtn_entry(model, pseudo_quant=args.pseudo_quant, wa_quant=wa_quant, q_group_size=args.w_group, w_bit=args.w_bit, a_bit=args.a_bit)
    else:
        raise NotImplementedError

    return model
