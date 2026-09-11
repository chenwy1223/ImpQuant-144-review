import torch


def visual_importance(confidence, jsd, alpha=0.1, vision_scale=1.0, eps=1e-2):
    """L_v / (-alpha log(confidence) + 1 / (JSD + eps))."""
    # Keep the logarithm's numerical floor independent of the JSD parameter.
    r = -alpha * confidence.float().clamp(min=1e-6, max=1).log() + (jsd.float() + eps).reciprocal()
    return vision_scale / r


def weighted_hessian(x, token_weights):
    """X P^2 X^T, with calibration tokens stored as rows in x."""
    if x.ndim != 2 or token_weights.shape != x.shape[:1]:
        raise ValueError('Expected x with shape (tokens, channels) and one weight per token')
    xp = x.float() * token_weights.float().unsqueeze(-1)
    if not xp.is_cuda:
        return xp.T @ xp
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('highest')
        return xp.T @ xp
    finally:
        torch.set_float32_matmul_precision(precision)
