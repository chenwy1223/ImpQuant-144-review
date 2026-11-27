"""Command-line arguments shared by calibration and evaluation."""
import argparse

METHODS = ('rtn', 'awq', 'smoothquant', 'gptq', 'mbq', 'impquant')


def parse_args(calibration=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='llava_onevision')
    parser.add_argument('--model_args', default='')
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--w_bit', type=int, default=4)
    parser.add_argument('--a_bit', type=int, default=16)
    parser.add_argument('--w_group', type=int, default=128)
    parser.add_argument('--scale_path')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--gptq_blocksize', type=int, default=128)
    parser.add_argument('--gptq_percdamp', type=float, default=0.01)
    parser.add_argument('--gptq_actorder', action='store_true')
    parser.add_argument('--impquant_text_scale', type=float, default=1.0)
    parser.add_argument('--impquant_vision_scale', type=float, default=1.0)
    parser.add_argument('--impquant_alpha', type=float, default=0.1)
    parser.add_argument('--impquant_jsd_eps', type=float, default=1e-2)
    parser.add_argument('--impquant_outlier_ratio', type=float, default=0.01)
    parser.add_argument('--impquant_probe_cache_path')
    parser.add_argument('--impquant_probe_source_path',
                        help='Reuse previously cached exact probe components during calibration')
    parser.add_argument('--impquant_probe_only', action='store_true')
    parser.add_argument('--impquant_skip_probe1', action='store_true',
                        help='Ablation: skip the visual repetition probe and use confidence=1')
    # uniform 走视觉等权路径，其他模式使用探针估计视觉权重。
    parser.add_argument('--impquant_probe2_mode', choices=('exact', 'grouped', 'selective', 'uniform'),
                        default='exact',
                        help='Probe-2 estimator or uniform visual weights with both visual probes disabled')
    parser.add_argument('--impquant_probe_group_h', type=int, default=2)
    parser.add_argument('--impquant_probe_group_w', type=int, default=2)
    parser.add_argument('--impquant_probe_refine_ratio', type=float, default=0.25)
    parser.add_argument('--impquant_probe_refine_strategy', choices=('top_jsd', 'random'),
                        default='top_jsd')
    parser.add_argument('--impquant_probe_refine_seed', type=int, default=2026)
    parser.add_argument('--seed', type=int, default=42)
    if calibration:
        parser.add_argument('--data_path', required=True)
        parser.add_argument('--image_folder', required=True)
        # 128 是通用默认值，calibrate_w3a16.py 显式传入 256。
        parser.add_argument('--n_samples', type=int, default=128)
        parser.add_argument('--calib_vision_micro_batch_size', type=int, default=0,
                            help='Split vision-tower calibration forwards by image; 0 keeps the original full batch')
        parser.add_argument('--calib_input_micro_batch_size', type=int, default=0,
                            help='Build multimodal calibration embeddings in sample chunks; 0 keeps one packed batch')
    else:
        parser.add_argument('--tasks', default='pope')
        parser.add_argument('--limit', type=int)
        parser.add_argument('--output_path', default='results.json')
        parser.add_argument('--save_samples', action='store_true',
                            help='Save lmms-eval per-sample records next to the aggregate JSON')
    args = parser.parse_args()
    if not (2 <= args.w_bit <= 16 and 2 <= args.a_bit <= 16):
        parser.error('Bit widths must be between 2 and 16')
    if args.w_group != -1 and args.w_group <= 0:
        parser.error('w_group must be -1 or positive')
    if args.method in ('awq', 'gptq') and args.a_bit != 16:
        parser.error('AWQ and GPTQ use a_bit=16')
    scale_optional = calibration and args.method == 'impquant' and args.impquant_probe_only
    if args.method != 'rtn' and not args.scale_path and not scale_optional:
        parser.error('This method requires --scale_path')
    if args.impquant_probe_only and (not calibration or args.method != 'impquant'):
        parser.error('impquant_probe_only is valid only for ImpQuant calibration')
    if args.impquant_probe_only and not args.impquant_probe_cache_path:
        parser.error('impquant_probe_only requires --impquant_probe_cache_path')
    if args.impquant_probe_source_path and (not calibration or args.method != 'impquant'):
        parser.error('impquant_probe_source_path is valid only for ImpQuant calibration')
    if args.impquant_skip_probe1 and args.method != 'impquant':
        parser.error('impquant_skip_probe1 is valid only for ImpQuant')
    if args.impquant_probe2_mode != 'exact' and args.method != 'impquant':
        parser.error('impquant_probe2_mode is valid only for ImpQuant')
    # uniform 等非 exact 模式要求同时关闭 Probe 1。
    if args.impquant_probe2_mode != 'exact' and not args.impquant_skip_probe1:
        parser.error('Non-exact Probe 2 modes require --impquant_skip_probe1')
    if min(args.impquant_probe_group_h, args.impquant_probe_group_w) <= 0:
        parser.error('ImpQuant probe group dimensions must be positive')
    if not 0 <= args.impquant_probe_refine_ratio <= 1:
        parser.error('impquant_probe_refine_ratio must be in [0, 1]')
    if not 0 <= args.alpha <= 1:
        parser.error('alpha must be in [0, 1]')
    if not 0 < args.impquant_outlier_ratio < 1:
        parser.error('impquant_outlier_ratio must be in (0, 1)')
    if min(args.impquant_text_scale, args.impquant_vision_scale, args.impquant_alpha) <= 0:
        parser.error('ImpQuant importance parameters must be positive')
    if not 0 < args.impquant_jsd_eps < float('inf'):
        parser.error('impquant_jsd_eps must be finite and positive')
    if args.gptq_blocksize <= 0 or args.gptq_percdamp <= 0:
        parser.error('GPTQ blocksize and percdamp must be positive')
    if args.batch_size <= 0 or (calibration and args.n_samples <= 0):
        parser.error('Batch size and sample count must be positive')
    if calibration and args.calib_vision_micro_batch_size < 0:
        parser.error('calib_vision_micro_batch_size must be non-negative')
    if calibration and args.calib_input_micro_batch_size < 0:
        parser.error('calib_input_micro_batch_size must be non-negative')
    args.gptq_groupsize = args.w_group
    args.run_process = calibration
    args.pseudo_quant = not calibration
    return args


def load_model(args):
    import random
    import numpy as np
    import torch
    from lmms_eval.models import get_model
    from qmllm.models import get_process_model
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    lm = get_model(args.model).create_from_arg_string(args.model_args,
            {'batch_size': args.batch_size, 'device': args.device})
    adapter = get_process_model(args.model)(lm._model, lm._tokenizer,
                                         getattr(lm, 'processor', None))
    return lm, adapter
