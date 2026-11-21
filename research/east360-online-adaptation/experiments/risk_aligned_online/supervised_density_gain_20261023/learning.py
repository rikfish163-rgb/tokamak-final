"""A label-fitted input coefficient, not a certified density-unit conversion.

Only one positive multiplicative coefficient is trained. The frozen source
normalizer, all other columns, observation masks and risk network stay fixed.
This is offline preparation, never claimed as intra-shot neural adaptation.
"""
import copy

import numpy as np
import torch


def apply_gain(values,available,log_gain):
    if (values.ndim!=3 or values.shape[-1]!=23 or values.shape!=available.shape
            or available.dtype!=torch.bool or log_gain.ndim!=0 or not torch.isfinite(log_gain)
            or not torch.isfinite(values[available]).all()):
        raise ValueError('finite observed B,T,23 inputs, boolean masks and a scalar finite log gain required')
    gain=log_gain.double().exp()
    converted=(values[:,:,12].double()*gain).to(values.dtype)
    if not torch.isfinite(converted[available[:,:,12]]).all() or not torch.isfinite(gain) or gain<=0:
        raise FloatingPointError('density gain produces unrepresentable observations')
    out=values.clone()
    out[:,:,12]=torch.where(available[:,:,12],converted,values[:,:,12])
    return out


def export_map(reference,gain):
    if not np.isfinite(gain) or gain<=0:raise ValueError('finite positive gain required')
    out=copy.deepcopy(reference)
    out['gain'][12]=float(gain);out['channels'][12]['gain']=float(gain)
    out['channels'][12]['status']='label_fitted_positive_gain'
    out.update(policy='label_fitted_density_gain',physical_calibration_certified=False,
        fitted_input_parameters=1,neural_parameter_updates=0,
        fitting='20 authorized historical TRAIN shots; balanced per-timepoint BCE, fixed risk network',
        initial_density_gain=reference['gain'][12])
    return out
