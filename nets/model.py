import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np


class DownConv(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 32, init_filters = 32, n_filters_per_depth=2,kernel_size=3, stride=1, padding=1, bias=True, groups=1):
        """
        A helper Module that performs either 2 or 3 convolutions and 1 MaxPool.
        A ReLU activation follows each convolution.
        """
        super(DownConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_filters = init_filters 
        self.n_filters_per_depth = n_filters_per_depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.groups = groups
        
        ins = self.in_channels
        outs = self.out_channels
        
        self.conv1 = nn.Conv2d(in_channels=ins, out_channels=outs, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups = self.groups)
        self.conv2 = nn.Conv2d(in_channels=outs, out_channels=outs, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups = self.groups)
        if(self.n_filters_per_depth==3):
            self.conv3 = nn.Conv2d(in_channels=outs, out_channels=outs, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups = self.groups)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)    
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if(self.n_filters_per_depth==3):
            x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x
    
    
class UpConv(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 32, init_filters = 32, n_filters_per_depth=2, kernel_size=3, stride=1, padding=1, bias=True, groups=1):
        """
        A helper Module that performs either 2 or 3 convolutions and 1 UpConvolution.
        A ReLU activation follows each convolution.
        """
        super(UpConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_filters = init_filters 
        self.n_filters_per_depth = n_filters_per_depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.groups = groups
        
        ins = self.in_channels
        outs = self.out_channels
        
        self.conv1 = nn.Conv2d(in_channels=ins, out_channels=outs, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups = self.groups)
        self.conv2 = nn.Conv2d(in_channels=outs, out_channels=outs, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups = self.groups)
        if(self.n_filters_per_depth==3):
            self.conv3 = nn.Conv2d(in_channels=outs, out_channels=outs, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups = self.groups)
        self.convtranspose = nn.ConvTranspose2d(in_channels=outs, out_channels=outs, kernel_size=2, stride=2)    
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if(self.n_filters_per_depth==3):
            x = F.relu(self.conv3(x))
        x = self.convtranspose(x)
        return x

    
class Encoder(nn.Module):
    def __init__(self, z_dim=4, in_channels = 1, init_filters = 32, n_filters_per_depth=2, n_depth=2,kernel_size=3, stride=1, padding=1, bias=True, groups=1):
        """
        Encoder pathway. It performs encoding operation and returns 
        latent space mean and log varaiance of the encoder distribution.
        """
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.in_channels = in_channels
        self.init_filters = init_filters 
        self.n_filters_per_depth = n_filters_per_depth
        self.n_depth = n_depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.groups = groups
        self.down_convs = []
        
        self.convmu = nn.Conv2d(in_channels=self.init_filters*(2**(self.n_depth-1)), out_channels=self.z_dim, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups = self.groups)
        self.convlogvar = nn.Conv2d(in_channels=self.init_filters*(2**(self.n_depth-1)), out_channels=self.z_dim, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups = self.groups)
        
        for i in range(n_depth):
            ins = self.in_channels if i==0 else outs
            outs = self.init_filters*(2**i)
            down_conv = DownConv(in_channels=ins, out_channels = outs, init_filters = self.init_filters, n_filters_per_depth=self.n_filters_per_depth)
            self.down_convs.append(down_conv)
     
        self.down_convs = nn.ModuleList(self.down_convs)
   
            
    def forward(self, x):
        # encoder pathway
        
        for i, module in enumerate(self.down_convs):
            x = module(x)
        
        mu = self.convmu(x)
        logvar = self.convlogvar(x)
        return mu, logvar
    
    
class Decoder(nn.Module):
    def __init__(self, z_dim=4, in_channels = 1, init_filters = 32, n_filters_per_depth=2, n_depth=2,kernel_size=3, stride=1, padding=1, bias=True, groups=1):
        """
        Decoder pathway. It performs decoding operation using the latent sample, 
        latent mean and latent log variance and returns the output image.
        """
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.in_channels = in_channels
        self.init_filters = init_filters 
        self.n_filters_per_depth = n_filters_per_depth
        self.n_depth = n_depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.groups = groups
        self.up_convs = []
        
        self.convrecon = nn.Conv2d(in_channels=self.init_filters, out_channels=1, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups=self.groups)

        for i in reversed(range(n_depth)):
            ins = self.z_dim if i==(n_depth-1) else outs
            outs = self.init_filters*(2**i)
            up_conv = UpConv(in_channels=ins, out_channels = outs, init_filters = self.init_filters, n_filters_per_depth=self.n_filters_per_depth)
            self.up_convs.append(up_conv)
            
        self.up_convs = nn.ModuleList(self.up_convs)
        
    def forward(self, x):
        for i, module in enumerate(self.up_convs):
            x = module(x)

        recon = self.convrecon(x)
        return recon

    
class VAE(nn.Module):
    def __init__(self, z_dim=4, in_channels = 1, init_filters = 32, n_filters_per_depth=2, n_depth=2,kernel_size=3, stride=1, padding=1, bias=True, groups=1):
        """
        Creates a DivNoising Variational Autoencoder model.
        It first makes an encoder object to get encoder distribution (latent space mean and log variance).
        Next the "reparametrize" method creates a sample from the encoder distribution.
        The latent sample is decoded using a decoder object.
        """
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.in_channels = in_channels
        self.init_filters = init_filters 
        self.n_filters_per_depth = n_filters_per_depth
        self.n_depth = n_depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.groups = groups
        
        
        self.encoder = Encoder(z_dim=self.z_dim, in_channels = self.in_channels, init_filters = self.init_filters, n_filters_per_depth=self.n_filters_per_depth, n_depth=self.n_depth, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups=self.groups)
               
        self.decoder = Decoder(z_dim=self.z_dim, in_channels = self.in_channels, init_filters = self.init_filters, n_filters_per_depth=self.n_filters_per_depth, n_depth=self.n_depth, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups=self.groups)
       
        self.cuda()

    def encode(self, x):
        """
        Performs encoding of input and returns latent space mean 
        and log variance of encoder distribution.
        """
        x_= (x-self.data_mean) / self.data_std
        return self.encoder(x_)
    
    def decode(self, z):
        """
        Performs decoding of given latent space sample 
        and returns the denormalized reconstructed image.
        """
        out = (self.decoder(z)* self.data_std)+self.data_mean 
        return out
        
    def reparameterize(self, mu, logvar):
        """
        Uses reparametrization trick as mentioned in the paper https://arxiv.org/abs/1312.6114
        to draw a sample from a normal distribution given its mean and log variance.
        """
        std = torch.exp(0.5*logvar)
        epsilon = torch.randn_like(mu)
        z = mu + epsilon*std
        return z