import pickle
import numpy as np
import torch

def to_numpy(ar):
    '''
    Converts a pytorch tensor to a numpy array
    '''
    if 'torch' in str(type(ar)):
        return ar.cpu().detach().numpy()
    else: return ar

def forward(model, x, g, y, maxis, use_vec, use_set, use_pers):
    '''
    This executes one forward pass and computes losses
    '''
    nloc = x.shape[2] * x.shape[3]
    # Compute forward pass
    if use_set and (use_vec or use_pers):
        pred = model(sets=g, vecs=x, maxis=maxis.reshape([-1, nloc]))
    elif use_set:
        pred = model(sets=g, maxis=maxis.reshape([-1, nloc]))
    elif use_vec or use_pers:
        pred = model(vecs=x, maxis=maxis.reshape([-1, nloc]))
    else:
        raise ValueError('need to use either vec or set or both to make a prediction')

    # Compute losses
    llk_loss, zero_prob = gp_penalized(y, pred[:, np.newaxis, :, :, 0], pred[:, np.newaxis, :, :, 1])
    loss = gp(y, pred[:, np.newaxis, :, :, 0], pred[:, np.newaxis, :, :, 1])
    return loss, to_numpy(llk_loss), to_numpy(zero_prob), to_numpy(pred)

def gp(raw_samples, xi, sigma):
    '''
    Computes NLK assuming generalized Pareto (GP) distribution
    NaNs are ignore for this computation
    Samples exceeding support of GP cause nans
    '''
    if 'numpy' in str(type(xi)):
        xi = torch.Tensor(xi)
    if 'numpy' in str(type(sigma)):
        sigma = torch.Tensor(sigma)
    if 'numpy' in str(type(raw_samples)):
        raw_samples = torch.Tensor(raw_samples)
    samples = torch.zeros_like(raw_samples) + raw_samples
    mask = ~torch.isnan(raw_samples)
    alt_mask = ~torch.isnan(xi)
    mask = mask & alt_mask
    samples[~mask] = 0
    out = torch.log(sigma) + (1 + 1/xi) * mask * torch.log(1 + xi * samples / sigma)
    return torch.mean(out[mask])

def theo_max(xi, sigma):
    '''
    Computes the upper bound of the generalized pareto distribution's support given shape and scale parameter
    '''
    out = np.zeros_like(xi) + 9999999999
    np.putmask(out, xi < 0, -sigma/xi)
    return out

def gp_penalized(raw_samples, xi, sigma):
    '''
    Computes NLK assuming generalized pareto distribution
    NaNs are ignored
    Samples exceeding support of GP are penalized
    '''
    if 'torch' in str(type(xi)):
        xi = xi.cpu().detach().numpy()
    if 'torch' in str(type(sigma)):
        sigma = sigma.cpu().detach().numpy()
    if 'torch' in str(type(raw_samples)):
        samples = raw_samples.cpu().detach().numpy()
    else:
        samples = raw_samples
    theo_maxes = theo_max(xi, sigma)
    nan_mask = ~np.isnan(samples)
    mask = (samples <= theo_maxes)
    cur_samples = samples
    log_thing = 1 + xi * cur_samples / sigma
    
    log_thing[(~mask) & (nan_mask)] = 1e-6
    out = np.log(sigma) + (1 + 1/xi) * np.log(log_thing)
    return np.nanmean(out[nan_mask]), np.sum( (~mask) & nan_mask ) / np.sum(nan_mask)
