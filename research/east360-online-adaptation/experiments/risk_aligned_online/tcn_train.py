"""Independent TCN23 source screening entry point.

This runner reuses the established flat23 data and source-evaluation helpers,
but constructs and trains TCN23 directly.  The default budget is a screening
budget and must not be described as a matched-superiority comparison.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import io23
from . import simple_data
from . import simple_train
from .tcn23 import CHECKPOINT_FORMAT, TCN23, TCNConfig, fit_forecaster


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--forecast-steps', type=int, default=256)
    parser.add_argument('--seed', type=int, default=131)
    parser.add_argument('--val-fpr-cap', type=float, default=.1)
    return parser


def _cpu_state(model):
    return {key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()}


def _public_metrics(measured):
    return {key: value for key, value in measured.items() if key != 'predictions'}


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if min(args.epochs, args.steps, args.batch_size) < 1:
        parser.error('epochs, steps, and batch-size must be positive')
    if args.forecast_steps < 0 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('forecast-steps must be nonnegative and lr must be positive')
    if not 0 < args.val_fpr_cap < .2:
        parser.error('source validation FPR cap must be below .20')
    if args.output.exists():
        raise FileExistsError('do not overwrite a prior experiment')
    if not torch.cuda.is_available():
        raise RuntimeError('local CUDA required for source training; run CPU tests separately')

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device('cuda')
    cfg = TCNConfig()

    roles = simple_data.source_roles()
    advanced = simple_data.advanced_times()
    train = simple_train.load_shots(roles['train'], advanced)
    val = simple_train.load_shots(roles['val'], advanced)
    scaler = simple_data.fit_source_scaler(train)
    model = TCN23(cfg, scaler['center'], scaler['scale']).to(device)
    forecast_ids = {id(parameter) for parameter in model.forecast_head.parameters()}
    risk_parameters = [parameter for parameter in model.parameters()
                       if id(parameter) not in forecast_ids]
    optimizer = torch.optim.AdamW(risk_parameters, lr=args.lr, weight_decay=.01)
    groups = simple_train.pools(train)

    code_files = ('tcn23.py', 'tcn_train.py', 'simple_data.py',
                  'simple_train.py', 'io23.py')
    config = dict(
        output=str(args.output.resolve()),
        format=CHECKPOINT_FORMAT,
        model='TCN23',
        model_config=cfg.to_dict(),
        epochs=args.epochs,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        forecast_steps=args.forecast_steps,
        seed=args.seed,
        val_fpr_cap=args.val_fpr_cap,
        normalization='TRAIN-only fit_source_scaler followed by source asinh',
        input='23 existing features plus 23 availability mask channels',
        feature_names=list(io23.RAW23),
        transform_contract={
            'J-TEXT_ip': 'multiply_1000',
            'EAST_MNM': 'unavailable',
            'normalization': 'source_train_asinh_iqr',
            'units': 'unknown_except_existing_ip_contract',
        },
        label='max(25ms,AdvancedTime)',
        source_roles={name: len(ids) for name, ids in roles.items()},
        source_role_ids=roles,
        selection='J-TEXT VAL160 selects epoch and threshold; TEST220 is read only after lock',
        budget_note='screening budget; not a matched-superiority comparison',
        source_gate='PENDING_REAL_TEST_RESULT_AND_EXTERNAL_REVIEW',
        evidence='DEVELOPMENT_ONLY',
        code_sha256={name: simple_train.sha(Path(__file__).with_name(name))
                     for name in code_files},
    )
    args.output.mkdir(parents=True)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))

    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for _ in range(args.steps):
            values, available, label, _, _ = simple_train.batch(
                train, groups, rng, args.batch_size, cfg, device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(
                model(values, available)['logit'], label.float())
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite source loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(risk_parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))

        measured = simple_train.select_threshold(
            simple_train.score(model, val, device), args.val_fpr_cap)
        record = dict(epoch=epoch, loss=float(np.mean(losses)),
                      validation=_public_metrics(measured))
        history.append(record)
        (args.output / 'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(record), flush=True)
        rank = simple_train.validation_rank(measured)
        if best is None or rank < best:
            best = rank
            source_risk = dict(
                format=CHECKPOINT_FORMAT,
                model=_cpu_state(model),
                model_config=cfg.to_dict(),
                source_center=scaler['center'],
                source_scale=scaler['scale'],
                threshold=measured['threshold'],
                validation=measured,
                forecast_trained=False,
                config=config,
                epoch=epoch,
            )
            torch.save(source_risk, args.output / 'source_risk.pt')

    selected = torch.load(args.output / 'source_risk.pt',
                          map_location=device, weights_only=False)
    model.load_state_dict(selected['model'])
    forecast_optimizer = torch.optim.AdamW(model.forecast_head.parameters(), lr=args.lr)
    forecast_losses = []
    for _ in range(args.forecast_steps):
        values, available, _, target, target_available = simple_train.batch(
            train, groups, rng, args.batch_size, cfg, device)
        forecast_losses.append(fit_forecaster(
            model, forecast_optimizer, values, available, target, target_available))

    selected['model'] = _cpu_state(model)
    selected['forecast_trained'] = bool(args.forecast_steps)
    selected['config'] = config
    torch.save(selected, args.output / 'source.pt')

    # TEST220 is intentionally touched only after the source epoch, threshold,
    # and the classifier checkpoint are locked.
    test = simple_train.load_shots(roles['test'], advanced)
    result = simple_train.metrics(
        simple_train.score(model, test, device), selected['threshold'])
    report = dict(
        format=CHECKPOINT_FORMAT,
        jtext=result,
        selected_epoch=selected['epoch'],
        threshold=selected['threshold'],
        forecast_trained=selected['forecast_trained'],
        forecast_train_loss=float(np.mean(forecast_losses)) if forecast_losses else None,
        source_gate='PENDING_REAL_TEST_RESULT_AND_EXTERNAL_REVIEW',
        evidence='DEVELOPMENT_ONLY',
        provenance=dict(
            checkpoint_sha256=simple_train.sha(args.output / 'source.pt'),
            selected_test_count=len(test),
            selection_source='J-TEXT VAL160',
            label='max(25ms,AdvancedTime)',
        ),
    )
    (args.output / 'test220.json').write_text(json.dumps(report, indent=2))
    hashes = dict(
        source_risk_sha256=simple_train.sha(args.output / 'source_risk.pt'),
        source_pt_sha256=simple_train.sha(args.output / 'source.pt'),
        test220_sha256=simple_train.sha(args.output / 'test220.json'),
        config_sha256=simple_train.sha(args.output / 'config.json'),
        code_sha256=config['code_sha256'],
    )
    (args.output / 'hashes.json').write_text(json.dumps(hashes, indent=2))
    print(json.dumps(dict(
        completed=True,
        selected_epoch=selected['epoch'],
        source_gate=report['source_gate'],
        **_public_metrics(result),
    )), flush=True)


if __name__ == '__main__':
    main()
