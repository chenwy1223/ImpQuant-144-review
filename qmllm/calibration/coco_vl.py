"""Load image–conversation calibration examples."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image


@torch.no_grad()
def get_multimodal_calib_dataset(data_path, image_folder, model, n_samples=128, seed=42):
    path = Path(data_path)
    with path.open() as stream:
        if path.suffix == '.jsonl':
            rows = [json.loads(line) for line in stream if line.strip()]
        elif path.suffix == '.json':
            rows = json.load(stream)
        else:
            raise ValueError('Calibration annotations must be JSON or JSONL')
    examples = []
    for row in rows:
        images = row.get('image', [])
        images = [images] if isinstance(images, str) else images
        if images and all('train2017' in image for image in images):
            examples.append(row)
    if len(examples) < n_samples:
        raise ValueError(f'Requested {n_samples} samples, but found {len(examples)} COCO train2017 examples')
    np.random.default_rng(seed).shuffle(examples)
    processed = []
    for row in examples[:n_samples]:
        names = row['image'] if isinstance(row['image'], list) else [row['image']]
        images = []
        for name in names:
            with Image.open(Path(image_folder) / name) as image:
                images.append(image.convert('RGB'))
        processed.append(model.preprocess_data(images, row))
    return model.generate_input(model.data_collator(processed))
