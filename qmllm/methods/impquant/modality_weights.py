"""Fixed image/text weighting for ImpQuant calibration."""
import math

import torch


VISION_TEXT_RATIO = 0.1


def fixed_modality_weights(raw_weights, valid, vision, text_scale=1., vision_scale=1.):
    if vision is None:
        raise ValueError('Fixed modality weighting requires vision_mask')
    visual = vision.bool() & valid
    text = ~vision.bool() & valid
    if not visual.any() or not text.any():
        raise ValueError('Calibration requires both visual and text tokens')
    rms_text = raw_weights[text].double().square().mean().sqrt().item()
    rms_visual = raw_weights[visual].double().square().mean().sqrt().item()
    if not (math.isfinite(rms_text) and math.isfinite(rms_visual) and min(rms_text, rms_visual) > 0):
        raise RuntimeError('Cannot determine L_t/L_v from invalid token importance')
    lt = text_scale / rms_text
    lv = vision_scale * math.sqrt(VISION_TEXT_RATIO) / rms_visual
    weights = torch.zeros_like(raw_weights)
    weights[text] = raw_weights[text] * lt
    weights[visual] = raw_weights[visual] * lv
    return weights, {'L_t': lt, 'L_v': lv,
                     'vision_text_ratio': VISION_TEXT_RATIO * (vision_scale / text_scale) ** 2}
