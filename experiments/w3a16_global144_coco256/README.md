# COCO-256 W3A16 校准与全局一阶选码

以 Qwen2.5-VL-7B-Instruct 为底座。先用 256 张 COCO 图文对校准 W3A16，再用任务监督梯度选择 144 个相邻整数码改动。

## 量化校准

`calibrate_w3a16.py` 使用 COCO train2017 单图图文清单和种子 42。权重为 3 位，激活保留 16 位，采用非对称 GPTQ，分组和块大小均为 128，请求 damping 0.1，关闭 actorder。ImpQuant 的视觉位置等权，文本权重取自模型最后一层 attention，随后按原方法做模态归一化和通道缩放。

```bash
python calibrate_w3a16.py \
  --model_args pretrained=/path/to/Qwen2.5-VL-7B-Instruct \
  --data_path /path/to/coco_train2017.json \
  --image_folder /path/to/coco_images \
  --scale_path /path/to/w3a16_uniform_coco256.pt
```

## 全局一阶选码

1. 在同一份量化底座上采集四组任务监督梯度。ScienceQA、A-OKVQA、TextVQA 各 16 题，OCR 组由 DocVQA 和 HME100K 各 8 题组成。选择题对正确选项计算首步交叉熵，开放题对正确答案 token 计算平均交叉熵。
2. 对第 7、11、15、19、23、27 层的 `v_proj`，从原整数码、scale 和零点构造合法的 `q ± 1`。要求原码在 1–6，且两侧移一格在 BF16 解码后均产生有限、非零的变化。每个模块按最佳邻码的一阶预测收益保留 128 个坐标及其两个方向。
3. 用 `g·Δw` 估计答案损失变化，`Δw` 取原量化网格解码到 BF16 后的实际变化。每组除以该组的平均答案损失，再四组等权求平均。六个模块统一按预测损失变化从小到大选满 144 个坐标，每个坐标至多选一个方向，每模块最多 96 码。同分时按层号、行、列、方向排序。
4. 用原 scale 和零点解码改动后的整数码，将替换张量写入缓存的对应 `v_proj` 权重。保留原分组、通道缩放和其他权重。

`selection.py` 提供可直接调用的计算核心。调用需要从同一份 W3A16 校准状态留存的六份原生 GPTQ 网格，以及同一状态上采集并按监督样本对齐的 BF16 梯度。校准入口保存量化缓存，原生码、scale、零点需在 GPTQ 求解时记录。

```python
from experiments.w3a16_global144_coco256.selection import repair_quantized_state, select_global_144

# grids[name] contains module, group_size, codes, scales and zeros.
# supervision[i] contains training_group and answer_nll.
# load_gradient(name, i) loads the native BF16 gradient for row i.
edits = select_global_144(grids, load_gradient, supervision)
updated_cache = repair_quantized_state(original_cache, grids, edits)
```
