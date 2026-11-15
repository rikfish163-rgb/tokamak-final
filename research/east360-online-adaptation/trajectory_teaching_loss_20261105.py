"""Fixed, per-shot point and shared-threshold trajectory teaching losses."""

import torch
import torch.nn.functional as F


def teaching_loss(logits, lead_ms, disrupted, valid, threshold_logit, mode):
    """Return one teaching loss for each padded shot in a batch.

    ``logits`` and ``lead_ms`` have shape [B, L]; ``disrupted`` is a bool
    tensor of shape [B], and ``valid`` is a bool tensor of shape [B, L].
    Inputs, including padding, are finite, and ``threshold_logit`` is scalar.
    Each shot must have a valid window, and each disrupted shot must have a
    valid lead strictly greater than 30 ms. There is no upper bound on this
    main-loss eligibility; the allowed-bin auxiliary may legitimately be absent.

    In trajectory mode the main term is the event negative log-likelihood
    under one logistic threshold jitter shared across an entire shot, rather
    than independently jittering each window. This is a training surrogate:
    it does not change the deployment threshold. The additional bin, early
    alarm, and safe top-quarter terms are auxiliary penalties. Windows are
    assumed to be the caller's chronological block representatives; candidate
    selection is not performed here.
    """
    if mode not in ("point", "trajectory"):
        raise ValueError("mode must be 'point' or 'trajectory'")
    if logits.ndim != 2 or lead_ms.shape != logits.shape or valid.shape != logits.shape:
        raise ValueError("logits, lead_ms, and valid must have matching [B, L] shapes")
    if disrupted.shape != (logits.shape[0],):
        raise ValueError("disrupted must have shape [B]")
    if not logits.is_floating_point():
        raise TypeError("logits must be floating point")
    if lead_ms.is_complex():
        raise TypeError("lead_ms must be real-valued")
    if valid.dtype != torch.bool or disrupted.dtype != torch.bool:
        raise TypeError("valid and disrupted must be bool tensors")
    if lead_ms.device != logits.device or valid.device != logits.device or disrupted.device != logits.device:
        raise ValueError("all input tensors must be on the same device")
    if isinstance(threshold_logit, torch.Tensor):
        if threshold_logit.ndim != 0 or threshold_logit.device != logits.device:
            raise ValueError("threshold_logit must be a scalar on the logits device")
    if logits.shape[1] == 0 or (~valid.any(dim=1)).any().item():
        raise ValueError("each shot must contain at least one valid window")

    # Preserve precise boundaries for narrow integers and bfloat16: their own
    # scalar comparisons can overflow or round 1500/3000 before comparing.
    leads = lead_ms if lead_ms.dtype in (torch.float32, torch.float64) else lead_ms.to(torch.float32)
    timely = valid & (leads > 30)
    if (disrupted & ~timely.any(dim=1)).any().item():
        raise ValueError("each disrupted shot needs a valid lead strictly greater than 30 ms")

    counts = valid.sum(dim=1)
    if mode == "point":
        targets = disrupted[:, None] & (leads >= 0) & (leads <= 125)
        # Mask before arithmetic so even large finite padding cannot affect it.
        observed_logits = logits.masked_fill(~valid, 0.0)
        point = F.binary_cross_entropy_with_logits(
            observed_logits - threshold_logit, targets.to(logits.dtype), reduction="none"
        )
        return point.masked_fill(~valid, 0.0).sum(dim=1) / counts

    temperature = 0.5
    observed_max = logits.masked_fill(~valid, -torch.inf).amax(dim=1)
    timely_present = timely.any(dim=1)
    timely_max = logits.masked_fill(~timely, -torch.inf).amax(dim=1)
    timely_max = torch.where(timely_present, timely_max, torch.zeros_like(timely_max))
    disrupted_main = F.softplus((threshold_logit - timely_max) / temperature)
    safe_main = F.softplus((observed_max - threshold_logit) / temperature)

    lower = leads.new_tensor([30, 125, 500, 1500])
    upper = leads.new_tensor([125, 500, 1500, 3000])
    in_bin = valid.unsqueeze(-1) & (leads.unsqueeze(-1) > lower) & (leads.unsqueeze(-1) <= upper)
    bin_present = in_bin.any(dim=1)
    bin_max = logits.unsqueeze(-1).masked_fill(~in_bin, -torch.inf).amax(dim=1)
    bin_max = torch.where(bin_present, bin_max, torch.zeros_like(bin_max))
    bin_count = bin_present.sum(dim=1)
    anchor = bin_max.masked_fill(~bin_present, -torch.inf).amax(dim=1)
    anchor = torch.where(bin_count > 0, anchor, torch.zeros_like(anchor))
    centered_bins = torch.where(bin_present, bin_max, anchor[:, None]) - anchor[:, None]
    scaled_bins = (centered_bins / temperature).masked_fill(~bin_present, -torch.inf)
    # A finite dummy is present only in rows with no bins. This prevents an
    # all-minus-infinity logsumexp and its undefined backward pass.
    dummy = torch.where(bin_count == 0, torch.zeros_like(observed_max), -torch.inf)
    normalized_bin_max = anchor + temperature * (
        torch.logsumexp(torch.cat((scaled_bins, dummy[:, None]), dim=1), dim=1)
        - bin_count.clamp_min(1).to(logits.dtype).log()
    )
    bin_aux = F.softplus((threshold_logit - normalized_bin_max) / temperature)
    bin_aux = torch.where(bin_count > 0, bin_aux, torch.zeros_like(bin_aux))

    early = valid & (leads > 3000)
    early_present = early.any(dim=1)
    early_max = logits.masked_fill(~early, -torch.inf).amax(dim=1)
    early_max = torch.where(early_present, early_max, torch.zeros_like(early_max))
    early_aux = torch.sigmoid((early_max - threshold_logit) / temperature)
    early_aux = torch.where(early_present, early_aux, torch.zeros_like(early_aux))

    top_count = (counts + 3) // 4
    sorted_logits = logits.masked_fill(~valid, -torch.inf).sort(dim=1, descending=True).values
    top_mask = torch.arange(logits.shape[1], device=logits.device)[None, :] < top_count[:, None]
    top_logits = sorted_logits.masked_fill(~top_mask, 0.0)
    top_terms = F.softplus((top_logits - threshold_logit) / temperature).masked_fill(~top_mask, 0.0)
    safe_aux = top_terms.sum(dim=1) / top_count

    return torch.where(
        disrupted,
        disrupted_main + 0.25 * bin_aux + 0.25 * early_aux,
        safe_main + 0.5 * safe_aux,
    )
