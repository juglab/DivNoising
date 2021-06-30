import torch
import os
import glob
import logging
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from divnoising import dataLoader, utils
from nets import lightningmodel
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

def create_dataloaders(x_train_tensor,x_val_tensor,batch_size):
    train_dataset = dataLoader.MyDataset(x_train_tensor,x_train_tensor)
    val_dataset = dataLoader.MyDataset(x_val_tensor,x_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader,val_loader

def create_model_and_train(basedir,data_mean,data_std,gaussian_noise_std,
                           noise_model,n_depth,max_epochs,logger,
                           checkpoint_callback,train_loader,val_loader,
                           kl_annealing, weights_summary):
    
    for filename in glob.glob(basedir+"/*"):
            os.remove(filename) 
    
    vae = lightningmodel.VAELightning(data_mean = data_mean,
                                      data_std = data_std,
                                      gaussian_noise_std = gaussian_noise_std,
                                      noise_model = noise_model,
                                      n_depth=n_depth,
                                      kl_annealing = kl_annealing)
        
    if torch.cuda.is_available():
        trainer = pl.Trainer(gpus=1, max_epochs=max_epochs, logger=logger,
                             callbacks=
                             [EarlyStopping(monitor='val_loss', min_delta=1e-6, 
                              patience = 100, verbose = True, mode='min'),checkpoint_callback], weights_summary=weights_summary)
    else:
        trainer = pl.Trainer(max_epochs=max_epochs, logger=logger,
                             callbacks=
                             [EarlyStopping(monitor='val_loss', min_delta=1e-6, 
                              patience = 100, verbose = True, mode='min'),checkpoint_callback], weights_summary=weights_summary)
    trainer.fit(vae, train_loader, val_loader)
    collapse_flag = trainer.should_stop
    return collapse_flag

def train_network(x_train_tensor, x_val_tensor, batch_size, data_mean, data_std, gaussian_noise_std, 
                  noise_model, n_depth, max_epochs, model_name, basedir, log_info=False):
    
    train_loader,val_loader = create_dataloaders(x_train_tensor, x_val_tensor, batch_size)
    collapse_flag = True
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    
    checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=basedir,
    filename=model_name+'_best',
    save_last=True,
    save_top_k=1,
    mode='min',)
    checkpoint_callback.CHECKPOINT_NAME_LAST = model_name+"_last"
    logger = TensorBoardLogger(basedir, name= "", version="", default_hp_metric=False)
    weights_summary="top" if log_info else None
    if not log_info:
        pl.utilities.distributed.log.setLevel(logging.ERROR)
    posterior_collapse_count = 0
    
    while collapse_flag and posterior_collapse_count<20:
        collapse_flag = create_model_and_train(basedir,data_mean,data_std,gaussian_noise_std,noise_model,
                                               n_depth,max_epochs,logger,checkpoint_callback,
                                               train_loader,val_loader,kl_annealing=False, weights_summary=weights_summary)
        if collapse_flag:
            posterior_collapse_count=posterior_collapse_count+1
        
    if collapse_flag:
        print("Posterior collapse limit reached, attempting training with KL annealing turned on!")
        while collapse_flag:
            collapse_flag = create_model_and_train(basedir,data_mean,data_std,gaussian_noise_std,noise_model,
                                               n_depth,max_epochs,logger,checkpoint_callback,
                                               train_loader,val_loader,kl_annealing=True, weights_summary=weights_summary)
       