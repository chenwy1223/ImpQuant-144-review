"""Fixed original Qwen image compatibility route; no model calls or sampling.

The same installed official helper is applied to every original document, then
the unchanged native processor handles patchification and text/image alignment.
"""
import hashlib

MIN_PIXELS = 200704
MAX_PIXELS = 1003520
INPUT_FIELDS = ['task', 'doc_id', 'source_index', 'target', 'metadata', 'prompt',
                'prompt_sha256', 'chat_prompt', 'chat_prompt_sha256', 'input_ids',
                'input_tensors', 'generation_kwargs', 'image_sha256', 'pixel_sha256',
                'raw_image_group_id', 'rgb_image_group_id', 'manifest_sha256']


def input_identity(record):
    return {key: record[key] for key in INPUT_FIELDS}


def canonical_inputs(processor, row, fetch_image_fn=None):
    """Return native inputs and exact conversion evidence for one original row."""
    from PIL import Image
    if fetch_image_fn is None:
        from qwen_vl_utils import fetch_image
        fetch_image_fn = fetch_image
    with Image.open(row['image']) as opened:
        original = opened.convert('RGB')
        original.load()
    assert list(original.size) == row['image_size'], 'Original image dimensions changed'
    original_rgb_sha = hashlib.sha256(original.tobytes()).hexdigest()
    assert original_rgb_sha == row['pixel_sha256'], 'Original decoded RGB changed'
    image = fetch_image_fn(dict(image=original, min_pixels=MIN_PIXELS,
                                max_pixels=MAX_PIXELS))
    assert image.mode == 'RGB' and min(image.size) >= 28
    assert image.width % 28 == image.height % 28 == 0
    messages = [dict(role='system', content='You are a helpful assistant.'),
                dict(role='user', content=[dict(type='image'),
                                          dict(type='text', text=row['prompt'])])]
    chat = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[chat], images=[image], videos=None,
                       padding=True, return_tensors='pt')
    metadata = dict(route='official_qwen_vl_utils_fetch_image_then_native_processor',
                    original_size=list(original.size), helper_resized_size=list(image.size),
                    original_short_side_below28=min(original.size) < 28,
                    original_rgb_sha256=original_rgb_sha,
                    helper_resized_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                    min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
                    interpolation='Installed official fetch_image implementation')
    return chat, inputs, metadata


def canonical_identity(row, manifest_sha, chat, inputs, generation, tensor_digest):
    tensor_records = {key: dict(sha256=tensor_digest(value), shape=list(value.shape),
                                dtype=str(value.dtype))
                      for key, value in inputs.items()}
    assert inputs['input_ids'].ndim == 2 and inputs['input_ids'].shape[0] == 1
    assert inputs['image_grid_thw'].shape == (1, 3)
    return dict(task=row['task'], doc_id=row['doc_id'], source_index=row['source_index'],
                target=row['target'], metadata=row['metadata'], prompt=row['prompt'],
                prompt_sha256=hashlib.sha256(row['prompt'].encode()).hexdigest(),
                chat_prompt=chat, chat_prompt_sha256=hashlib.sha256(chat.encode()).hexdigest(),
                input_ids=inputs['input_ids'][0].tolist(), input_tensors=tensor_records,
                generation_kwargs=generation, image_sha256=row['sha256'],
                pixel_sha256=row['pixel_sha256'], raw_image_group_id=row['raw_image_group_id'],
                rgb_image_group_id=row['rgb_image_group_id'], manifest_sha256=manifest_sha)
