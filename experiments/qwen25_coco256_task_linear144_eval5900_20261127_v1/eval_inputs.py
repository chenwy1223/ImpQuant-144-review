"""Build the same deterministic Qwen input for each side of a paired question."""
import hashlib
from pathlib import Path

from PIL import Image, ImageOps
from processor_compat import MAX_PIXELS, MIN_PIXELS, canonical_inputs
from native_contracts import tensor_digest

# Bound the total vision-token load for multi-image MMMU documents on one 3090.
MMMU_DOCUMENT_PIXELS = 1605632
PROCESSOR = None


def processor():
    global PROCESSOR
    if PROCESSOR is None:
        from transformers import AutoProcessor
        from common import MODEL
        PROCESSOR = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True,
                                                   min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    return PROCESSOR


def raw_digest(row):
    return [im["sha256"] for im in row["images"]]


def generation_kwargs(row, proc):
    limit = 16 if row["answer_kind"]=="choice" or row["task"]=="mmmu_val" else 128
    return dict(max_new_tokens=limit, do_sample=False, num_beams=1, use_cache=True,
                eos_token_id=proc.tokenizer.eos_token_id, pad_token_id=proc.tokenizer.pad_token_id)


def prefixes(row, proc):
    from common import sha
    from qwen_vl_utils import fetch_image

    refs = row["images"]
    assert 1 <= len(refs) <= 5
    for im in refs:
        assert sha(Path(im["image"])) == im["sha256"], row["id"]
    if len(refs)==1 and not row["extreme_aspect"]:
        _, tensors, _ = canonical_inputs(proc, {**refs[0], "prompt":row["prompt"]})
    else:
        pictures = []
        max_each=min(MAX_PIXELS, (MMMU_DOCUMENT_PIXELS//len(refs)//784)*784)
        assert max_each >= MIN_PIXELS
        for item in refs:
            with Image.open(item["image"]) as opened:
                im=opened.convert("RGB")
                im.load()
            assert list(im.size)==item["image_size"]
            assert hashlib.sha256(im.tobytes()).hexdigest()==item["pixel_sha256"]
            # Native fetch_image requires an aspect ratio below 200.
            if max(im.size) >= 200*min(im.size):
                w,h=im.size
                w_pad,h_pad=max(0,(h+198)//199-w), max(0,(w+198)//199-h)
                im=ImageOps.expand(im,(w_pad//2,h_pad//2,w_pad-w_pad//2,h_pad-h_pad//2),fill="white")
            pictures.append(fetch_image(dict(image=im,min_pixels=MIN_PIXELS,max_pixels=max_each)))
        messages=[dict(role="system",content="You are a helpful assistant."),
                  dict(role="user",content=[dict(type="image") for _ in pictures]+
                       [dict(type="text",text=row["prompt"])])]
        chat=proc.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        tensors=proc(text=[chat],images=pictures,videos=None,padding=True,return_tensors="pt")
    prefix = {k:v.cpu().contiguous() for k,v in tensors.items()}
    assert set(prefix)=={"input_ids","attention_mask","pixel_values","image_grid_thw"}
    assert prefix["input_ids"].shape[0]==1 and len(prefix["image_grid_thw"])==len(refs)
    vision_tokens=sum(int(t)*int(h)*int(w)//4 for t,h,w in prefix["image_grid_thw"].tolist())
    assert vision_tokens <= MMMU_DOCUMENT_PIXELS//784, (row["id"],vision_tokens)
    return prefix


def tensor_records(prefix):
    return {k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=tensor_digest(v)) for k,v in prefix.items()}


def decode(ids, proc, task):
    x=proc.batch_decode([ids],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
    result=x.split(proc.tokenizer.decode(proc.tokenizer.eos_token_id))[0]
    if task=="vizwiz_vqa_val":
        result=result.split("ASSISTANT:",1)[0]
    return result
