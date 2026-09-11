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
    parser.add_argument('--seed', type=int, default=42)
    if calibration:
        parser.add_argument('--data_path', required=True)
        parser.add_argument('--image_folder', required=True)
        parser.add_argument('--n_samples', type=int, default=128)
    else:
        parser.add_argument('--tasks', default='pope')
        parser.add_argument('--limit', type=int)
        parser.add_argument('--output_path', default='results.json')
    args = parser.parse_args()
    if not (2 <= args.w_bit <= 16 and 2 <= args.a_bit <= 16):
        parser.error('Bit widths must be between 2 and 16')
    if args.w_group != -1 and args.w_group <= 0:
        parser.error('w_group must be -1 or positive')
    if args.method in ('awq', 'gptq') and args.a_bit != 16:
        parser.error('AWQ and GPTQ use a_bit=16')
    if args.method != 'rtn' and not args.scale_path:
        parser.error('This method requires --scale_path')
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
