"""Evaluate a quantized model with lmms-eval."""
import json
from pathlib import Path
from types import SimpleNamespace
from qmllm.cli import parse_args, load_model


def main():
    args = parse_args()
    from lmms_eval import evaluator
    from qmllm.quantization.quant_wrapper import qwrapper
    lm, model = load_model(args)
    qwrapper(model, None, None, args)
    output = Path(args.output_path)
    artifact_dir = output.parent / 'lmms_artifacts' / output.stem
    artifact_dir.mkdir(parents=True, exist_ok=True)
    results = evaluator.simple_evaluate(model=args.model, lm=lm,
        model_args=args.model_args, tasks=args.tasks.split(','),
        batch_size=args.batch_size, device=args.device, limit=args.limit,
        random_seed=args.seed, numpy_random_seed=args.seed, torch_random_seed=args.seed,
        cli_args=SimpleNamespace(output_path=str(artifact_dir)))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results['results'], indent=2, default=str) + '\n')
    if args.save_samples:
        samples_output = output.with_name(output.stem + '_samples.json')
        samples_output.write_text(
            json.dumps(results.get('samples', {}), indent=2, default=str) + '\n'
        )
        print(samples_output)
    print(output)


if __name__ == '__main__':
    main()
