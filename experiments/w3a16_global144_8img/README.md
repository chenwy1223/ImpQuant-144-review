# 256 图校准底座上的 W3A16 全局一阶 144 码

在 Qwen2.5-VL-7B-Instruct 的同一份 256 张 COCO 图校准的 matched-uniform ImpQuant/GPTQ W3A16 底座上，只改动 144 个整数码。分组大小为 128，沿用原 scale、zero、输入缩放和其余模型参数。

| 任务 | 配对题数 | 原量化 | 改 144 码 | 差值（百分点） |
| --- | ---: | ---: | ---: | ---: |
| VizWiz | 1,000 | 66.01% | 65.87% | −0.14 |
| ScienceQA-IMG | 1,000 | 83.70% | 84.50% | +0.80 |
| MMBench EN | 1,000 | 83.90% | 84.40% | +0.50 |
| TextVQA | 1,000 | 82.88% | 83.34% | +0.46 |
| OCRBench | 1,000 | 83.80% | 83.50% | −0.30 |
| MMMU Val | 900 | 41.78% | 43.44% | +1.67 |
| 六项等权均分 | 5,900 | 73.678% | 74.176% | **+0.498** |

均分差的配对 95% bootstrap 区间为 [+0.155, +0.850] 个百分点。VizWiz 和 TextVQA 使用共识软分。这是探索性评测，部分题目此前用于开发。MMMU 多图题的续跑使用了在真实样本上逐位核对过的分块视觉注意力。

## 选码和应用

- `repair.py` 的 `select_global_144` 按冻结的一阶损失变化 `g·Δw` 从小到大选择。`Δw` 是合法相邻码经原网格反量化后的实际 BF16 步长。候选来自六层 `v_proj`，共有 768 个坐标、1,536 个方向。同坐标只能选一次，每层最多 96 码。同分时依层号、行、列、方向排序，`−1` 先于 `+1`。
- `patch.json` 是实际评测用的 144 个改码，按选择顺序记录。前 96 码与上一轮相同，追加 48 码。第 7、11、15、19 层分别改 87、7、47、3 码。第 23、27 层各 0 码。
- `build_replacements` 使用原始整数码及冻结的 FP32 scale、zero 生成四份 BF16 张量，并核对整数码和张量指纹。归档的全模型逐张量审计覆盖 1,121 个张量，涉及差异的键均为上述四层的 `v_proj.weight`。

## 复核方式

使用归档的 `ef_projection.pt`、原生网格和原量化 checkpoint。`patch.json` 记录输入投影、checkpoint、评测补丁和状态的 SHA-256。本目录提取了实际实验的选码与补丁核心，评测沿用原实验流水线。

```python
import json
from pathlib import Path

import torch
from repair import build_replacements, select_global_144, tensor_digest

patch = json.loads(Path('patch.json').read_text(encoding='utf-8'))
# projection and grids are the archived CPU objects. model is the pinned W3A16 base.
assert select_global_144(projection['a'], projection['moves']) == patch['selected_indices']
replacements = build_replacements(grids, patch)
targets = {name: model.get_submodule(name).weight for name in replacements}
for name, target in targets.items():
    assert tensor_digest(target) == patch['tensor_receipts'][name]['baseline_weight_sha256']
with torch.no_grad():
    for name, target in targets.items():
        target.copy_(replacements[name])
```

选码与原始补丁来自 `qwen25_global144_eval1024_20261120_v1`。表中六任务结果来自 `qwen25_impquant6_linear144_eval5900_20261126_v3`。
