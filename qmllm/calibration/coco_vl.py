"""Load image–conversation calibration examples."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _move_tensor_dict(mapping, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in mapping.items()
    }


def _merge_generated_batches(batches, device):
    max_length = max(inputs['inputs_embeds'].shape[1] for inputs, _ in batches)
    merged_inputs = []
    merged_kwargs = {}
    for inputs, kwargs in batches:
        embeddings = inputs['inputs_embeds']
        pad_length = max_length - embeddings.shape[1]
        merged_inputs.append(F.pad(embeddings, (0, 0, pad_length, 0), value=0))
        for key, value in kwargs.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f'Unsupported non-tensor calibration kwarg: {key}')
            pad_value = -100 if key == 'labels' else 0
            padded = F.pad(value, (pad_length, 0), value=pad_value)
            merged_kwargs.setdefault(key, []).append(padded)
    prompt_inputs = {'inputs_embeds': torch.cat(merged_inputs, dim=0).to(device)}
    prompt_kwargs = {}
    for key, values in merged_kwargs.items():
        batch_dim = 1 if key == 'position_ids' and values[0].ndim == 3 else 0
        prompt_kwargs[key] = torch.cat(values, dim=batch_dim).to(device)
    return prompt_inputs, prompt_kwargs


@torch.no_grad()
def get_multimodal_calib_dataset(data_path, image_folder, model, n_samples=128, seed=42,
                                 input_micro_batch_size=0):
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
    if input_micro_batch_size <= 0 or input_micro_batch_size >= len(processed):
        return model.generate_input(model.data_collator(processed))

    batches = []
    device = next(model.model.parameters()).device
    for start in range(0, len(processed), input_micro_batch_size):
        chunk = processed[start:start + input_micro_batch_size]
        prompt_inputs, prompt_kwargs = model.generate_input(model.data_collator(chunk))
        batches.append((
            _move_tensor_dict(prompt_inputs, 'cpu'),
            _move_tensor_dict(prompt_kwargs, 'cpu'),
        ))
        del prompt_inputs, prompt_kwargs
        torch.cuda.empty_cache()
    return _merge_generated_batches(batches, device)
