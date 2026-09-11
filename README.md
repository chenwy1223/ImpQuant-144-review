# ImpQuant

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -c requirements.txt -e ./3rdparty/lmms-eval
pip install -c requirements.txt -e './3rdparty/LLaVA-NeXT[standalone]'
pip install -e .
```

## Calibration data

Prepare a JSON array of image–conversation pairs and the COCO `train2017` images:

```json
[
  {
    "image": "train2017/000000000009.jpg",
    "conversations": [
      {"from": "human", "value": "<image>\nDescribe this image."},
      {"from": "gpt", "value": "The image shows a collection of bowls."}
    ]
  }
]
```

## Calibrate a method

Choose `--method` from `impquant`, `mbq`, `awq`, `smoothquant`, or `gptq`:

```bash
python main_quant.py \
  --method impquant \
  --model llava_onevision --model_args pretrained=/path/to/model \
  --w_bit 3 --a_bit 16 \
  --data_path /path/to/calibration.json --image_folder /path/to/coco \
  --scale_path scale_cache/impquant_w3a16.pt
```

## Evaluate a method

```bash
python main.py \
  --method impquant \
  --model llava_onevision --model_args pretrained=/path/to/model \
  --w_bit 3 --a_bit 16 \
  --scale_path scale_cache/impquant_w3a16.pt \
  --tasks task_name --output_path results/impquant_w3a16.json
```

Set `--tasks` to one task name or a comma-separated list, such as `scienceqa_img,textvqa_val`. Available tasks also include `pope`, `ocrbench`, `mmmu_val`, `vizwiz_vqa_val`, and `mmbench_en_dev`.

RTN runs directly without calibration:

```bash
python main.py \
  --method rtn \
  --model llava_onevision --model_args pretrained=/path/to/model \
  --w_bit 3 --a_bit 16 \
  --tasks task_name --output_path results/rtn_w3a16.json
```

## Q-VLM

Run [Q-VLM](https://github.com/ChangyuanWang17/QVLM) in its own environment:

```bash
conda create -n qvlm python=3.10 -y
conda activate qvlm
git clone https://github.com/ChangyuanWang17/QVLM.git ../QVLM
cd ../QVLM
git checkout 191bdc9a833b7ed273c234d9f7b69ceb51d33b6f
pip install -e .
pip uninstall -y bitsandbytes
(cd custom_bitsandbytes && python setup.py install)
```

Specify the LLaVA-v1.3 checkpoint and the ScienceQA annotation and image paths:

```bash
python -m llava.eval.model_vqa_science \
  --model-path /path/to/llava-v1.3-checkpoint \
  --question-file /path/to/scienceqa/llava_test_QCM-LEPA.json \
  --image-folder /path/to/scienceqa/images/test \
  --question-file-calibrate /path/to/scienceqa/llava_train_QCM-LEPA.json \
  --image-folder-calibrate /path/to/scienceqa/images/train \
  --answers-file results/qvlm_sqa.jsonl \
  --conv-mode llava_v1 --load-4bit
```
