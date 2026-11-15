"""Consume each direct-forecast horizon once, when its real target arrives.

The parameter layout is identical to DiagnosticMemory. All four forecast
horizons are issued together. No predicted signal ever becomes a training
target, and no past prediction is recomputed for retrospective scoring.
"""
import torch

from ..delayed_recurrent_memory_20261025.model import ARMS, HORIZON, DiagnosticMemory


class ProgressiveDiagnosticMemory(DiagnosticMemory):
    def trace(self, means, complete, *, mode, reference_norm=None, return_before=False):
        if (mode not in ('off',) + ARMS or means.ndim != 2 or means.shape[1] != 23
                or complete.shape != means.shape or complete.dtype != torch.bool):
            raise ValueError('declared mode and observed diagnostic patches required')
        if mode in ('local', 'hold', 'reverse') and (
                reference_norm is None or reference_norm.shape != means.shape[:1]):
            raise ValueError('controls require main-arm norm at every patch')
        features = torch.cat((means * complete, complete.to(means)), -1)
        keys = torch.cat((self.key(features).tanh(), means.new_ones(len(means), 1)), -1)
        # Observation counts are diagnostics, not a source of update decisions.
        pairs = [complete[h:] & complete[:-h] for h in range(1, HORIZON + 1)]
        horizon_targets = [int(p.any(-1).sum()) for p in pairs]
        target_scalars = sum(int(p.sum()) for p in pairs)
        write_mask = torch.zeros(len(means), dtype=torch.bool, device=means.device)
        for h, pair in enumerate(pairs, 1):
            write_mask[h:] |= pair.any(-1)
        writes = write_mask.detach().cpu().tolist()
        state = self.prior
        deltas, forecasts, norms, zeros, before = [], [], [], [], []
        first_target = first_mask = None
        for j in range(len(means)):
            if return_before:
                # Offline teaching compares the same observed token with and
                # without its latest signal-driven write; no outcome enters here.
                before.append(keys[j] @ (state-self.prior))
            zero = torch.zeros((), dtype=torch.bool, device=means.device)
            if j:
                count = min(j, HORIZON)
                origins = torch.arange(j-1, j-count-1, -1, device=means.device)
                target = torch.cat((means[j]-means[origins], means.new_zeros(HORIZON-count, 23)))
                mask = torch.cat((complete[j] & complete[origins],
                                  complete.new_zeros(HORIZON-count, 23)))
                origin_keys = torch.cat((keys[origins], keys.new_zeros(HORIZON-count, 9)))
                if first_target is None:
                    first_target, first_mask = target, mask
                if mode != 'off' and writes[j]:
                    if mode == 'hold':
                        target = torch.zeros_like(target)
                    if mode == 'reverse':
                        # Swap horizon destinations, never substitute future data.
                        target, mask, origin_keys = target.flip(0), mask.flip(0), origin_keys.flip(0)
                    current = self.prior if mode == 'local' else state
                    columns = current.reshape(9, HORIZON, 23).permute(1, 0, 2)
                    error = ((origin_keys[:, :, None]*columns).sum(1)-target)*mask
                    gradient = (origin_keys[:, :, None]*error[:, None, :]
                                / (1+origin_keys.square().sum(-1))[:, None, None])
                    # Disjoint horizon columns: one combined update, one cap.
                    candidate = current-self.eta*gradient.permute(1, 0, 2).reshape(9, 92)
                    change = candidate-self.prior
                    state = self.prior+change/torch.linalg.vector_norm(change).clamp_min(1.)
                    if mode in ('local', 'hold', 'reverse'):
                        change = state-self.prior
                        norm = torch.linalg.vector_norm(change)
                        desired = reference_norm[j].detach()
                        zero = (norm == 0) & (desired > 0)
                        # Report zero-direction cases; do not manufacture a direction.
                        state = self.prior+change*desired/norm.clamp_min(torch.finfo(norm.dtype).tiny)
            forecasts.append((keys[j] @ state).reshape(HORIZON, 23))
            deltas.append(keys[j] @ (state-self.prior))
            norms.append(torch.linalg.vector_norm(state-self.prior))
            zeros.append(zero)
        forecast = torch.stack(forecasts) if forecasts else means.new_zeros(0, HORIZON, 23)
        # Same full-horizon diagnostic as the delayed rule; partial tail targets
        # are counted above but cannot fabricate a full-horizon forecast metric.
        if len(means) > HORIZON:
            target = torch.stack([means[h:len(means)-HORIZON+h]-means[:-HORIZON]
                                  for h in range(1, HORIZON+1)], 1)
            mask = torch.stack([complete[h:len(means)-HORIZON+h] & complete[:-HORIZON]
                                for h in range(1, HORIZON+1)], 1)
            loss = ((forecast[:-HORIZON]-target).square()*mask).sum()/mask.sum().clamp_min(1)
        else:
            loss = means.sum()*0
        result = dict(delta=torch.stack(deltas) if deltas else means.new_zeros(0, 92),
                    forecast=forecast, forecast_loss=loss,
                    norm=torch.stack(norms) if norms else means.new_zeros(0),
                    zero_direction=torch.stack(zeros) if zeros else complete.new_zeros(0),
                    updates=sum(writes) if mode != 'off' else 0,
                    first_target=first_target, first_mask=first_mask,
                    horizon_targets=horizon_targets, target_scalars=target_scalars)
        if return_before:
            result['before_delta'] = torch.stack(before) if before else means.new_zeros(0, 92)
        return result
