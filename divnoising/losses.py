import torch.optim as optim
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np
import matplotlib.pyplot as plt
# import torchvision
import torch.nn.functional as F
import time
import datetime
from divnoising import utils

def lossFunctionKLD(mu, logvar):
    """Compute KL divergence loss. 
    Parameters
    ----------
    mu: Tensor
        Latent space mean of encoder distribution.
    logvar: Tensor
        Latent space log variance of encoder distribution.
    """
    kl_error = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_error

def recoLossGaussian(predicted_s, x, gaussian_noise_std, data_std):
    """
    Compute reconstruction loss for a Gaussian noise model.  
    This is essentially the MSE loss with a factor depending on the standard deviation.
    Parameters
    ----------
    predicted_s: Tensor
        Predicted signal by DivNoising decoder.
    x: Tensor
        Noisy observation image.
    gaussian_noise_std: float
        Standard deviation of Gaussian noise.
    data_std: float
        Standard deviation of training and validation data combined (used for normailzation).
    """
    reconstruction_error =  torch.mean((predicted_s-x)**2) / (2.0* (gaussian_noise_std/data_std)**2 )
    return reconstruction_error

def recoLoss(predicted_s, x, data_mean, data_std, noiseModel):
    """Compute reconstruction loss for an arbitrary noise model. 
    Parameters
    ----------
    predicted_s: Tensor
        Predicted signal by DivNoising decoder.
    x: Tensor
        Noisy observation image.
    data_mean: float
        Mean of training and validation data combined (used for normailzation).
    data_std: float
        Standard deviation of training and validation data combined (used for normailzation).
    device: GPU device
        torch cuda device
    """
    predicted_s_denormalized = predicted_s * data_std + data_mean
    x_denormalized = x * data_std + data_mean
    predicted_s_cloned = predicted_s_denormalized
    predicted_s_reduced = predicted_s_cloned.permute(1,0,2,3)
    
    x_cloned = x_denormalized
    x_cloned = x_cloned.permute(1,0,2,3)
    x_reduced = x_cloned[0,...]
    
    likelihoods=noiseModel.likelihood(x_reduced,predicted_s_reduced)
    log_likelihoods=torch.log(likelihoods)

    # Sum over pixels and batch
    reconstruction_error= -torch.mean( log_likelihoods ) 
    return reconstruction_error
    
def loss_fn(predicted_s, x, mu, logvar, gaussian_noise_std, data_mean, data_std, noiseModel):
    """Compute DivNoising loss. 
    Parameters
    ----------
    predicted_s: Tensor
        Predicted signal by DivNoising decoder.
    x: Tensor
        Noisy observation image.
    mu: Tensor
        Latent space mean of encoder distribution.
    logvar: Tensor
        Latent space logvar of encoder distribution.
    gaussian_noise_std: float
        Standard deviation of Gaussian noise (required when using Gaussian reconstruction loss).
    data_mean: float
        Mean of training and validation data combined (used for normailzation).
    data_std: float
        Standard deviation of training and validation data combined (used for normailzation).
    device: GPU device
        torch cuda device
    noiseModel: NoiseModel object
        Distribution of noisy pixel values corresponding to clean signal (required when using general reconstruction loss).
    """
    kl_loss = lossFunctionKLD(mu, logvar)
    
    if noiseModel is not None:
        reconstruction_loss = recoLoss(predicted_s, x, data_mean, data_std, noiseModel)
    else:
        reconstruction_loss = recoLossGaussian(predicted_s, x, gaussian_noise_std, data_std)
    #print(float(x.numel()))
    return reconstruction_loss, kl_loss /float(x.numel())