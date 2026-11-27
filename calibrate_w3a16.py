"""Qwen2.5-VL W3A16 calibration with the existing model and COCO path arguments."""
import sys

from main_quant import main


if __name__ == "__main__":
    sys.argv[1:1] = [
        "--model", "qwen2_5_vl",
        # 跳过两个视觉探针，使用视觉等权，沿用 ImpQuant 的文本权重与通道缩放。
        "--method", "impquant", "--impquant_skip_probe1", "--impquant_probe2_mode", "uniform",
        # W3A16 使用 3 位权重码和 16 位激活，GPTQ 分组大小为 128。
        "--w_bit", "3", "--a_bit", "16", "--w_group", "128",
        "--gptq_blocksize", "128", "--gptq_percdamp", "0.1",
        # 用 256 张 COCO 图及其描述校准，种子 42 固定采样顺序。
        "--n_samples", "256", "--seed", "42",
        # 输入构造和视觉塔前向都按单条样本分批，控制峰值内存和显存。
        "--calib_input_micro_batch_size", "1", "--calib_vision_micro_batch_size", "1",
    ]
    main()
