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
import torchvision
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

def recoLoss(predicted_s, x, data_mean, data_std, device, noiseModel):
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
    
def loss_fn(predicted_s, x, mu, logvar, gaussian_noise_std, data_mean, data_std, device, noiseModel):
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
        reconstruction_loss = recoLoss(predicted_s, x, data_mean, data_std, device, noiseModel)
    else:
        reconstruction_loss = recoLossGaussian(predicted_s, x, gaussian_noise_std, data_std)
    #print(float(x.numel()))
    return reconstruction_loss, kl_loss /float(x.numel())

def trainNetwork(net, train_loader, val_loader, device,
                 data_mean,data_std,
                 model_name,
                 directory_path='.',
                 n_epochs=1100,
                 batch_size=32,
                 lr=0.001,
                 val_loss_patience = 300,
                 kl_annealing = False,
                 kl_start = 2, # The number of epochs at which KL loss should be included
                 kl_annealtime = 5, # number of epochs over which KL scaling is increased from 0 to 1
                 kl_min=1e-5,
                 gaussian_noise_std = None,
                 noiseModel = None
                ):
    """Compute KL divergence loss. 
    Parameters
    ----------
    net: VAE object
        DivNoising model.
    train_loader: PyTorch data loader
        Data loader for training set.
    val_loader: PyTorch data loader
        Data loader for validation set.
    device: GPU device
        torch cuda device
    data_mean: float
        Mean of training and validation data combined.
    data_std: float
        Standard deviation of training and validation data combined.
    model_name: String
        Name of DivNoising model with which to save weights.
    directory_path: String
        Path where the DivNoising weights to be saved.
    n_epochs: int
        Number of epochs to train the model for.
    batch_size: int
        Batch size for training
    lr: float
        Learning rate
    val_loss_patience: int
        Number of epoochs after which training should be terminated if validation loss doesn't improve by 1e-6.
    kl_annealing: boolean
        Use KL annealing for training or not.
    kl_start: int
        Epoch from which to start KL annealing.
    kl_annealtime: int
        epochs until which KL annealing to be performed.
    kl_min=1e-5: float
        If the KL loss drops below this value, we consider this a posterior collapse and abort training.
    noiseModel: NoiseModel object
        Distribution of noisy pixel values corresponding to clean signal.
        If this is None a Gaussian noise model is used (specify 'gaussian_noise_std').
    gaussian_noise_std: float
        standard deviation of gaussian noise (required when 'noiseModel' is None).
    
    Returns
    ----------
    loss_train_history:
        a 1D np array containing the training loss
        'None' if aborted due to posterior collapse 
    reconstruction_loss_train_history:
        a 1D np array containing the training reconstruction loss
        'None' if aborted due to posterior collapse
    kl_loss_train_history:
        a 1D np array containing the training KL loss
        'None' if aborted due to posterior collapse
    loss_val_history
        a 1D np array containing the validation loss loss
        'None' if aborted due to posterior collapse
    """
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           'min',
                                                           patience=30,
                                                           factor=0.5,
                                                           min_lr=1e-12,
                                                           verbose=True)
    
    net.data_mean = torch.Tensor([data_mean]).to(device)
    net.data_std = torch.Tensor([data_std]).to(device)
    
    assert(data_std is not None)
    if noiseModel is None:
        assert(gaussian_noise_std is not None)
    else:
        assert(gaussian_noise_std is None)
        assert(data_mean is not None) 
    
    loss_train_history = []
    reconstruction_loss_train_history = []
    kl_loss_train_history = []
    loss_val_history = []
    
    patience_ = 0
    
    seconds_last = time.time()
    for epoch in range(n_epochs):
        
        running_training_loss = []
        running_reconstruction_loss = []
        running_kl_loss = []
        
        if(kl_annealing==True):
            #calculate weight
            kl_weight = (epoch-kl_start) * (1.0/kl_annealtime)
            # clamp to [0,1]
            kl_weight = min(max(0.0,kl_weight),1.0)
        else:
            kl_weight = 1.0
        
        
        for x, _ in train_loader:
            x = x.cuda()
            x = (x-net.data_mean) / net.data_std
            mu, logvar = net.encoder(x)
            z = net.reparameterize(mu, logvar)
            recon = net.decoder(z)
            reconstruction_loss, kl_loss = loss_fn(recon, x, 
                                                   mu,
                                                   logvar,
                                                   gaussian_noise_std,
                                                   net.data_mean,
                                                   net.data_std,
                                                   device,
                                                   noiseModel)
            
            # check for posterior collapse
            if kl_loss < kl_min:
                print('postersior collapse: aborting')
                return None, None, None, None
            
            
            loss = reconstruction_loss + kl_weight*kl_loss
            
            running_training_loss.append(loss.item())
            running_reconstruction_loss.append(reconstruction_loss.item())
            running_kl_loss.append(kl_loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        ### Print training losses
        to_print = "Epoch[{}/{}] Training Loss: {:.3f} Reconstruction Loss: {:.3f} KL Loss: {:.3f}"
        to_print = to_print.format(epoch+1,
                                  n_epochs, 
                                  np.mean(running_training_loss),
                                  np.mean(running_reconstruction_loss),
                                  np.mean(running_kl_loss))
        print(to_print)
        print('kl_weight:',kl_weight)
        print('saving',directory_path+model_name+"last_vae.net")
        torch.save(net, directory_path+model_name+"last_vae.net")
        
        ### Save training losses 
        loss_train_history.append(np.mean(running_training_loss))
        reconstruction_loss_train_history.append(np.mean(running_reconstruction_loss))
        kl_loss_train_history.append(np.mean(running_kl_loss))
        np.save(directory_path+"train_loss.npy", np.array(loss_train_history))
        np.save(directory_path+"train_reco_loss.npy", np.array(reconstruction_loss_train_history))
        np.save(directory_path+"train_kl_loss.npy", np.array(kl_loss_train_history))
        
        ### Validation step
        running_validation_loss = []
        with torch.no_grad():
            for i, (x, _) in enumerate(val_loader):
                x = x.cuda()
                x = (x-net.data_mean) / net.data_std
                mu, logvar = net.encoder(x)
                z = net.reparameterize(mu, logvar)
                recon = net.decoder(z)
                val_reconstruction_loss, val_kl_loss = loss_fn(recon,
                                                               x, 
                                                               mu,
                                                               logvar, 
                                                               gaussian_noise_std, 
                                                               net.data_mean,
                                                               net.data_std,
                                                               device, noiseModel)
                val_loss = val_reconstruction_loss+val_kl_loss
                running_validation_loss.append(val_loss)

        normalizer_val = len(val_loader.dataset)
        total_epoch_loss_val = torch.mean(torch.stack(running_validation_loss))
        scheduler.step(total_epoch_loss_val)
        
        ### Save validation losses      
        loss_val_history.append(total_epoch_loss_val.item())
        np.save(directory_path+"val_loss.npy", np.array(loss_val_history))

        if total_epoch_loss_val.item() < 1e-6 + np.min(loss_val_history):
            patience_ = 0
            print('saving',directory_path+model_name+"best_vae.net")
            torch.save(net, directory_path+model_name+"best_vae.net")
        else:
            patience_ +=1

        print("Patience:", patience_,
              "Validation Loss:", total_epoch_loss_val.item(),
              "Min validation loss:", np.min(loss_val_history))
        
        seconds=time.time()
        secondsElapsed=np.float(seconds-seconds_last)
        seconds_last=seconds
        remainingEps=n_epochs-(epoch+1)
        estRemainSeconds=(secondsElapsed)*(remainingEps)
        estRemainSecondsInt=int(secondsElapsed)*(remainingEps)
        print('Time for epoch: '+ str(int(secondsElapsed))+ 'seconds')
        
        print('Est remaining time: '+
              str(datetime.timedelta(seconds= estRemainSecondsInt)) +
              ' or ' +
              str(estRemainSecondsInt)+ 
              ' seconds')
        
        print("----------------------------------------", flush=True)
        
        
        if patience_ == val_loss_patience:
            break
            
    return loss_train_history, reconstruction_loss_train_history, kl_loss_train_history, loss_val_history