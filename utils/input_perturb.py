import torch


def _build_random_mask(x, mask_ratio):
    return torch.rand_like(x) < mask_ratio


def _build_block_mask(x, mask_ratio, block_len):
    batch, seq_len, _ = x.shape
    target_steps = max(1, int(round(seq_len * mask_ratio)))
    block_len = max(1, min(int(block_len), seq_len))
    mask = torch.zeros_like(x, dtype=torch.bool)

    for batch_idx in range(batch):
        covered = 0
        while covered < target_steps:
            cur_block = min(block_len, target_steps - covered)
            max_start = max(1, seq_len - cur_block + 1)
            start = int(torch.randint(0, max_start, (1,), device=x.device).item())
            end = min(seq_len, start + cur_block)
            mask[batch_idx, start:end, :] = True
            covered += cur_block

    return mask


def build_input_mask(x, mask_ratio=0.0, mask_mode='random', mask_block_len=16):
    if mask_ratio <= 0:
        return torch.zeros_like(x, dtype=torch.bool)
    if mask_ratio >= 1:
        raise ValueError(f"mask_ratio must be in [0, 1), got {mask_ratio}")

    if mask_mode == 'random':
        return _build_random_mask(x, mask_ratio)
    if mask_mode == 'block':
        return _build_block_mask(x, mask_ratio, mask_block_len)

    raise ValueError(f"Unsupported mask_mode: {mask_mode}")


def apply_input_perturbation(
    x,
    noise_std=0.0,
    mask_ratio=0.0,
    mask_mode='random',
    mask_block_len=16,
    mask_value=0.0,
):
    out = x
    if noise_std > 0:
        out = out + torch.randn_like(out) * float(noise_std)

    if mask_ratio > 0:
        mask = build_input_mask(out, mask_ratio=mask_ratio, mask_mode=mask_mode, mask_block_len=mask_block_len)
        fill = torch.full_like(out, float(mask_value))
        out = torch.where(mask, fill, out)

    return out
