import numpy as np
import torch
from scipy.stats import entropy
import utils
from tqdm import tqdm
import math
import models as mds
import itertools


def total_uncertainty_variance(probs):
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().numpy()
    mean = np.mean(probs, axis=2)
    t_u = np.sum(mean*(1-mean), axis=1)
    return t_u

def aleatoric_uncertainty_variance(probs):
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().numpy()
    a_u = np.mean(np.sum(probs*(1-probs), axis=1), axis=1)
    return a_u

def epistemic_uncertainty_variance(probs):
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().numpy()
    mean = np.mean(probs, axis=2, keepdims=1)
    e_u = np.mean(np.sum(probs*(probs-mean), axis=1), axis=1)
    return e_u


def total_uncertainty_entropy(probs):
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().numpy()
    t_u = entropy(np.mean(probs, axis=2), axis=1, base=2) / np.log2(probs.shape[1])
    return t_u


def epistemic_uncertainty_entropy(probs):
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().numpy()
    mean_probs = np.mean(probs, axis=2)
    mean_probs = np.repeat(np.expand_dims(mean_probs, 2), repeats=probs.shape[2], axis=2)
    mean_probs = np.clip(mean_probs, 1e-25, 1)
    probs = np.clip(probs, 1e-25, 1)
    e_u = entropy(probs, mean_probs, axis=1, base=2) / np.log2(probs.shape[1])
    e_u = np.mean(e_u, axis=1)
    return e_u


def aleatoric_uncertainty_entropy(probs):
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().numpy()
    a_u = entropy(probs, axis=1, base=2) / np.log2(probs.shape[1])
    a_u = np.mean(a_u, axis=1)
    return a_u


def remove_rejected(y_pred, y_true, reject_portion, uncertainties):
    if reject_portion == 0:
        return y_pred, y_true
    num = int(y_pred.shape[0] * reject_portion)
    indices = np.argsort(uncertainties)
    y_pred = y_pred[indices]
    y_true = y_true[indices]
    y_pred = y_pred[:-num]
    y_true = y_true[:-num]
    return y_pred, y_true


def remove_random(y_pred, y_true, reject_portion):
    if reject_portion == 0:
        return y_pred, y_true
    num = int(y_pred.shape[0] * reject_portion)
    indices = np.random.permutation(y_pred.shape[0])
    y_pred = y_pred[indices]
    y_true = y_true[indices]
    y_pred = y_pred[:-num]
    y_true = y_true[:-num]
    return y_pred, y_true







