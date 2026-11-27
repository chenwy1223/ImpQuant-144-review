# 256 张 COCO 校准基线上的全局一阶 144 码修复

这份提交记录完成评测的 Qwen2.5-VL-7B W3A16 实现。选码使用四组任务监督的归一化梯度估计 −gΔw。六个 v_proj 共用 144 码预算，沿原 scale、零点和分组移动一步。

- supervision.py 在新基线上计算 64 道监督题的梯度。
- selection.py 统一排名并封存 144 个合法改动。
- grid_assembly.py 由原生整数网格生成修改后的权重。
- science.py 用逐题配对的方式评测修复前后。
- audit.py 独立重算选码、配对得分和统计结果。

运行配置通过 IMPQUANT_ROOT、IMPQUANT_MODEL_DIR、IMPQUANT_GPU_INDEX 和 IMPQUANT_GPU_UUID 提供。prepare.py 读取本地 source_pins.json，用于绑定实验所需的历史量化状态、六份原生网格、监督数据及评测记录。提交包含实验源文件和协议。完整输入与执行证据保存在实验环境中。
