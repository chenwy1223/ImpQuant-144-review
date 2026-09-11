"""Calibrate a quantization method and save its state."""
from qmllm.cli import parse_args, load_model


def main():
    args = parse_args(calibration=True)
    from qmllm.calibration.coco_vl import get_multimodal_calib_dataset
    from qmllm.quantization.quant_wrapper import qwrapper
    _, model = load_model(args)
    inputs, kwargs = get_multimodal_calib_dataset(args.data_path, args.image_folder,
                                                 model, n_samples=args.n_samples, seed=args.seed)
    qwrapper(model, inputs, kwargs, args)


if __name__ == '__main__':
    main()
