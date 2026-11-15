"""Source-keyed, locally activated feature updates INSIDE the last FFN.

Keys are fitted only to frozen J-TEXT TRAIN activations. No risk head, final
logit correction, target statistics, or future outcome enters the router.
The disposable and persistent 8x64 matrices shift hidden features before the
original FFN residual/final LayerNorm; both original heads see the result.
Uniform routing is a same-parameter non-local mechanism control.
"""
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .ffn_intra_20261005.model import FfnModel
from .persistent_ffn import PersistentFfnModel
from .redesign_20261004.source_replay import (
    DEFAULT_PARENT, PARENT_SHA256, load_teacher, load_source_bank, sha256)


def fit_router(activations, clusters=8):
    """Deterministic spherical k-means, with SOURCE-only hidden coordinates."""
    if (type(clusters) is not int or clusters != 8 or activations.ndim != 2
            or activations.shape[1] != 128 or len(activations) < clusters
            or not activations.is_floating_point() or not torch.isfinite(activations).all()):
        raise ValueError('finite [N>=8,128] activations and eight keys required')
    x = activations.detach().float()
    center = x.median(0).values
    quantiles = torch.quantile(x, x.new_tensor([.25, .75]), dim=0)
    # The positive floor is in hidden-activation coordinates, not a physical
    # unit calibration. It prevents constant neurons defining infinite distance.
    scale = (quantiles[1] - quantiles[0]).clamp_min(1e-4)
    vectors = F.normalize((x-center)/scale, dim=-1, eps=1e-12)
    if not vectors.square().sum(-1).gt(.5).any():
        raise ValueError('source hidden variation is required')
    chosen = [int(((x-center)/scale).square().sum(-1).argmax())]
    distance = 1 - vectors @ vectors[chosen[0]]
    for _ in range(1, clusters):
        index = int(distance.argmax())
        if float(distance[index]) < 1e-6:
            raise ValueError('eight distinguishable source states require more variation')
        chosen.append(index)
        distance = torch.minimum(distance, 1-vectors @ vectors[index])
    keys = vectors[chosen].clone()
    for _ in range(20):
        membership = (vectors @ keys.T).argmax(-1)
        updated = keys.clone()
        for k in range(clusters):
            group = vectors[membership == k]
            # Empty cells retain their declared key: the k-means objective has
            # no observations to update that cell, and the action is explicit.
            if len(group):
                candidate = group.mean(0)
                if candidate.norm() > 1e-12: updated[k] = F.normalize(candidate, dim=0)
        keys = updated
    return dict(center=center, scale=scale, keys=keys)


def capture_hidden(base, x, available):
    """Fixed original source path; no hook remains after success or failure."""
    captured = []
    def read(module, inputs, output): captured.append(inputs[0].detach())
    hook = base.encoder.layers[-1].linear2.register_forward_hook(read)
    try:
        with torch.enable_grad():
            fast = base.initial_fast(len(x)).detach().requires_grad_(True)
            base(x, available, fast)
    finally:
        hook.remove()
    if len(captured) != 1: raise ValueError('exactly one last-FFN activation batch required')
    return captured[0]


def fit_source_artifact(base, bank):
    hidden = capture_hidden(base, bank['x'].to(base.center),
                            bank['available'].to(base.center.device))
    patch_valid = bank['available'].reshape(len(hidden), 16, -1).any(-1)
    valid = torch.cat((patch_valid, torch.ones(len(hidden), 1, dtype=torch.bool)), 1)
    vectors = hidden[valid.to(hidden.device)]
    router = {k: v.cpu() for k, v in fit_router(vectors).items()}
    return dict(router=router, source_parent_sha256=PARENT_SHA256,
        source_bank_content_sha256=bank['bank_sha256'],
        source_role='TRAIN', target_records_accessed=0, classifier_retrained=False,
        label_used_for_key_fitting=False, clusters=8, temperature=.1, top_k=2,
        hidden_vectors=len(vectors), source_shot_ids=bank['shotids'].tolist(),
        location='last Transformer linear2 input; frozen source activation coordinates',
        initialization='farthest source states; 20 spherical k-means iterations')


def load_router(path, *, bank_sha256):
    artifact = torch.load(path, map_location='cpu', weights_only=True)
    if (artifact['source_parent_sha256'] != PARENT_SHA256
            or artifact['source_bank_content_sha256'] != bank_sha256
            or artifact['source_role'] != 'TRAIN' or artifact['target_records_accessed'] != 0
            or artifact['classifier_retrained'] or artifact['label_used_for_key_fitting']
            or (artifact['clusters'], artifact['temperature'], artifact['top_k']) != (8, .1, 2)):
        raise ValueError('fixed source-only routing artifact required')
    return artifact


class RoutedFfnModel(FfnModel):
    adaptation_site = 'source-keyed hidden feature increment inside last Transformer FFN'
    summarize_fast = PersistentFfnModel.summarize_fast

    def __init__(self, base, router, *, routing='local'):
        if routing not in ('local', 'uniform'): raise ValueError('known routing required')
        super().__init__(base)
        self.routing = routing
        expected = dict(center=(128,), scale=(128,), keys=(8, 128))
        for name, shape in expected.items():
            value = router[name].detach().clone().to(self.center)
            if value.shape != shape or not torch.isfinite(value).all():
                raise ValueError('finite source routing coordinates of declared shape required')
            self.register_buffer('router_'+name, value)
        if not (self.router_scale > 0).all(): raise ValueError('positive source hidden scale required')
        if not torch.allclose(self.router_keys.norm(dim=-1), self.center.new_ones(8), atol=1e-5):
            raise ValueError('unit source keys required')
        self.temporary_dim = 512
        self.memory = nn.Parameter(self.center.new_zeros(8, 64))

    def initial_fast(self, batch):
        if type(batch) is not int or batch < 0: raise ValueError('nonnegative batch required')
        return self.center.new_zeros(batch, self.temporary_dim)

    def persistent_parameters(self): return iter((self.memory,))

    def forecast_loss(self, prediction, truth, available):
        return self.base.forecast_loss(prediction, truth, available)

    def route(self, hidden):
        if hidden.shape[-1] != 128 or not torch.isfinite(hidden).all():
            raise ValueError('finite last-FFN hidden states required')
        if self.routing == 'uniform': return hidden.new_full((*hidden.shape[:-1], 8), 1/8)
        vectors = F.normalize((hidden.detach()-self.router_center)/self.router_scale,
                              dim=-1, eps=1e-12)
        similarities = vectors @ self.router_keys.T
        # Stable ordering makes equal similarity deterministic. Two active keys
        # are a fixed mechanism hypothesis, not chosen from EAST performance.
        indices = torch.argsort(similarities, descending=True, stable=True)[..., :2]
        scores = similarities.gather(-1, indices) / .1
        return torch.zeros_like(similarities).scatter(-1, indices, scores.softmax(-1))

    def forward(self, values, available, fast=None):
        if torch.is_inference_mode_enabled(): raise RuntimeError('inference_mode changes the declared source path')
        if values.ndim != 3 or not len(values) or values.shape[1:] != (128, 23):
            raise ValueError('nonempty W128 flat23 batch required')
        fast = self.initial_fast(len(values)) if fast is None else fast
        if (fast.shape != (len(values), 512) or fast.device != self.center.device
                or fast.dtype != self.center.dtype or not torch.isfinite(fast).all()):
            raise ValueError('finite source-device temporary state required')
        if not torch.equal(fast, fast[:1].expand_as(fast)):
            raise ValueError('one same shot temporary state is required')
        enabled = torch.is_grad_enabled()
        with torch.enable_grad():
            working = self.memory + fast[0].reshape(8, 64)
            def adapt(module, inputs, output):
                return output + self.route(inputs[0]) @ working
            hook = self.base.encoder.layers[-1].linear2.register_forward_hook(adapt)
            try:
                source_fast = self.base.initial_fast(len(values)).detach().requires_grad_(True)
                result = self.base(values, available, source_fast)
            finally:
                hook.remove()
        return result if enabled else {k: v.detach() for k, v in result.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('new artifact path required')
    torch.set_num_threads(4)
    bank_path = Path(__file__).parent/'runs/redesign_source_bank_20261004.pt'
    bank = load_source_bank(bank_path, expected_parent_sha256=PARENT_SHA256)
    base = load_teacher(DEFAULT_PARENT).to('cuda')
    original = {k: v.detach().clone() for k, v in base.state_dict().items()}
    artifact = fit_source_artifact(base, bank)
    if (sha256(DEFAULT_PARENT) != PARENT_SHA256
            or any(not torch.equal(v, base.state_dict()[k]) for k, v in original.items())):
        raise ValueError('original source changed while fitting fixed routing keys')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle: torch.save(artifact, handle)
    print(dict(output=str(args.output), source_role='TRAIN', hidden_vectors=artifact['hidden_vectors'],
               target_records_accessed=0, classifier_retrained=False), flush=True)


if __name__ == '__main__': main()
