"""Qwen2.5-VL W3A16 calibration with the existing model and COCO path arguments."""
import sys

from main_quant import main


if __name__ == "__main__":
    sys.argv[1:1] = [
        "--model", "qwen2_5_vl",
        "--method", "gptq",
        "--w_bit", "3", "--a_bit", "16", "--w_group", "128",
        "--gptq_blocksize", "128", "--gptq_percdamp", "0.1",
        "--n_samples", "256", "--seed", "42",
        "--calib_input_micro_batch_size", "1", "--calib_vision_micro_batch_size", "1",
    ]
    main()
