import torch
dtype = torch.float
import matplotlib.pyplot as plt
import numpy as np
import pickle
from torch.distributions import normal
from scipy.stats import norm
from tifffile import imread

def fastShuffle(series, num):
    length = series.shape[0]
    for i in range(num):
        series = series[np.random.permutation(length),:]
    return series

import divnoising.histNoiseModel

class GaussianMixtureNoiseModel:
    """The GaussianMixtureNoiseModel class describes a noise model which is parameterized as a mixture of gaussians.
       If you would like to initialize a new object from scratch, then set `params`= None and specify the other parameters as keyword arguments. If you are instead loading a model, use only `params`.


            Parameters
            ----------
            **kwargs: keyworded, variable-length argument dictionary.
            Arguments include:
                min_signal : float
                    Minimum signal intensity expected in the image.
                max_signal : float
                    Maximum signal intensity expected in the image.
                path: string
                    Path to the directory where the trained noise model (*.npz) is saved in the `train` method.
                weight : array
                    A [3*n_gaussian, n_coeff] sized array containing the values of the weights describing the noise model.
                    Each gaussian contributes three parameters (mean, standard deviation and weight), hence the number of rows in `weight` are 3*n_gaussian.
                    If `weight=None`, the weight array is initialized using the `min_signal` and `max_signal` parameters.
                n_gaussian: int
                    Number of gaussians.
                n_coeff: int
                    Number of coefficients to describe the functional relationship between gaussian parameters and the signal.
                    2 implies a linear relationship, 3 implies a quadratic relationship and so on.
                device: device
                    GPU device.
                min_sigma: int
                    All values of sigma (`standard deviation`) below min_sigma are clamped to become equal to min_sigma.
                params: dictionary
                    Use `params` if one wishes to load a model with trained weights. 
                    While initializing a new object of the class `GaussianMixtureNoiseModel` from scratch, set this to `None`.

            Example
            -------
            >>> model = GaussianMixtureNoiseModel(min_signal = 484.85, max_signal = 3235.01, path='../../models/', weight = None, n_gaussian = 3, n_coeff = 2, min_sigma = 50, device = torch.device("cuda:0"))
    """

    def __init__(self, **kwargs):
        if(kwargs.get('params') is None):
            weight=kwargs.get('weight')
            n_gaussian=kwargs.get('n_gaussian')
            n_coeff=kwargs.get('n_coeff')
            min_signal=kwargs.get('min_signal')
            max_signal=kwargs.get('max_signal')
            self.device=kwargs.get('device')
            self.path=kwargs.get('path')
            self.min_sigma=kwargs.get('min_sigma')
            if (weight is None):
                weight = np.random.randn(n_gaussian * 3, n_coeff)
                weight[n_gaussian:2 * n_gaussian, 1] = np.log(max_signal - min_signal)
                weight = torch.from_numpy(weight.astype(np.float32)).float().to(self.device)
                weight.requires_grad = True
            self.n_gaussian = weight.shape[0] // 3
            self.n_coeff = weight.shape[1]
            self.weight = weight
            self.min_signal = torch.Tensor([min_signal]).to(self.device)
            self.max_signal = torch.Tensor([max_signal]).to(self.device)
            self.tol = torch.Tensor([1e-10]).to(self.device)
        else:
            params=kwargs.get('params')
            self.device=kwargs.get('device')
            
            
            self.min_signal=torch.Tensor(params['min_signal']).to(self.device)
            self.max_signal=torch.Tensor(params['max_signal']).to(self.device)
            
            self.weight=torch.Tensor(params['trained_weight']).to(self.device)
            self.min_sigma=np.float(params['min_sigma'])
            self.n_gaussian=self.weight.shape[0]//3
            self.n_coeff=self.weight.shape[1]
            self.tol=torch.Tensor([1e-10]).to(self.device)
            self.min_signal=torch.Tensor([self.min_signal]).to(self.device)
            self.max_signal=torch.Tensor([self.max_signal]).to(self.device)

    


    def polynomialRegressor(self, weightParams, signals):
        """Combines `weightParams` and signal `signals` to regress for the gaussian parameter values.

                Parameters
                ----------
                weightParams : torch.cuda.FloatTensor
                    Corresponds to specific rows of the `self.weight`

                signals : torch.cuda.FloatTensor
                    Signals
                Returns
                -------
                value : torch.cuda.FloatTensor
                    Corresponds to either of mean, standard deviation or weight, evaluated at `signals`
        """
        value=0
        for i in range(weightParams.shape[0]):
            value += weightParams[i] * (((signals - self.min_signal) / (self.max_signal - self.min_signal)) ** i);
        return value
        
   
    def normalDens(self, x, m_= 0.0, std_= None):
        """Evaluates the normal probability density at `x` given the mean `m` and standard deviation `std`.

                Parameters
                ----------
                x: torch.cuda.FloatTensor
                    Observations
                m_: torch.cuda.FloatTensor
                    Mean
                std_: torch.cuda.FloatTensor
                    Standard-deviation
                Returns
                -------
                tmp: torch.cuda.FloatTensor
                    Normal probability density of `x` given `m_` and `std_`

        """

        tmp=-((x-m_)**2) 
        tmp=tmp / (2.0*std_*std_)
        tmp= torch.exp(tmp )
        tmp= tmp/ torch.sqrt( (2.0*np.pi)*std_*std_)
        return tmp

    def likelihood(self, observations, signals):
        """Evaluates the likelihood of observations given the signals and the corresponding gaussian parameters.

                Parameters
                ----------
                observations : torch.cuda.FloatTensor
                    Noisy observations
                signals : torch.cuda.FloatTensor
                    Underlying signals
                Returns
                -------
                value :p + self.tol
                    Likelihood of observations given the signals and the GMM noise model

        """
        gaussianParameters=self.getGaussianParameters(signals)
        p=0
        for gaussian in range(self.n_gaussian):
            p+=self.normalDens(observations, gaussianParameters[gaussian],
                               gaussianParameters[self.n_gaussian+gaussian])*gaussianParameters[2*self.n_gaussian+gaussian]
        return p+self.tol
    
    def getGaussianParameters(self, signals):
        """Returns the noise model for given signals

                Parameters
                ----------
                signals : torch.cuda.FloatTensor
                    Underlying signals
                Returns
                -------
                noiseModel: list of torch.cuda.FloatTensor
                    Contains a list of `mu`, `sigma` and `alpha` for the `signals`

        """
        noiseModel = []
        mu = []
        sigma = []
        alpha = []
        kernels = self.weight.shape[0]//3
        for num in range(kernels):
            mu.append(self.polynomialRegressor(self.weight[num, :], signals))
            
            sigmaTemp=self.polynomialRegressor(torch.exp(self.weight[kernels+num, :]), signals)
            sigmaTemp = torch.clamp(sigmaTemp, min = self.min_sigma)
            sigma.append(torch.sqrt(sigmaTemp))
            alpha.append(torch.exp(self.polynomialRegressor(self.weight[2*kernels+num, :], signals) + self.tol))

        sum_alpha = 0
        for al in range(kernels):
            sum_alpha = alpha[al]+sum_alpha
        for ker in range(kernels):
            alpha[ker]=alpha[ker]/sum_alpha
            
        sum_means = 0
        for ker in range(kernels):
            sum_means = alpha[ker]*mu[ker]+sum_means

        mu_shifted=[]
        for ker in range(kernels):
            mu[ker] = mu[ker] - sum_means + signals
        
        for i in range(kernels):
            noiseModel.append(mu[i])
        for j in range(kernels):
            noiseModel.append(sigma[j])
        for k in range(kernels):
            noiseModel.append(alpha[k])

        return noiseModel
    
    def getSignalObservationPairs(self, signal, observation, lowerClip, upperClip):
        """Returns the Signal-Observation pixel intensities as a two-column array

                Parameters
                ----------
                signal : numpy array
                    Clean Signal Data
                observation: numpy array
                    Noisy observation Data
                lowerClip: float
                    Lower percentile bound for clipping.
                upperClip: float
                    Upper percentile bound for clipping.

                Returns
                -------
                noiseModel: list of torch floats
                    Contains a list of `mu`, `sigma` and `alpha` for the `signals`

        """
        lb = np.percentile(signal, lowerClip)
        ub = np.percentile(signal, upperClip)
        stepsize=observation[0].size
        n_observations=observation.shape[0]
        n_signals=signal.shape[0]
        sig_obs_pairs= np.zeros((n_observations*stepsize,2))
        
        for i in range(n_observations):
            j = i//(n_observations//n_signals)
            sig_obs_pairs[stepsize*i:stepsize*(i+1), 0] = signal[j].ravel()
            sig_obs_pairs[stepsize*i:stepsize*(i+1), 1] = observation[i].ravel()
        sig_obs_pairs = sig_obs_pairs[ (sig_obs_pairs[:,0]>lb) & (sig_obs_pairs[:,0]<ub)]
        return fastShuffle(sig_obs_pairs, 2)
        
  

    def train(self, signal, observation, learning_rate=1e-1, batchSize=250000, n_epochs=2000, name= 'GMMNoiseModel.npz', lowerClip=0, upperClip=100):
        """Training to learn the noise model from signal - observation pairs.

                Parameters
                ----------
                signal: numpy array
                    Clean Signal Data
                observation: numpy array
                    Noisy Observation Data
                learning_rate: float
                    Learning rate. Default = 1e-1.
                batchSize: int
                    Nini-batch size. Default = 250000.
                n_epochs: int
                    Number of epochs. Default = 2000.
                name: string

                    Model name. Default is `GMMNoiseModel`. This model after being trained is saved at the location `path`.

                lowerClip : int
                    Lower percentile for clipping. Default is 0.
                upperClip : int
                    Upper percentile for clipping. Default is 100.
                    
                    
        """
        sig_obs_pairs=self.getSignalObservationPairs(signal, observation, lowerClip, upperClip)
        counter=0
        optimizer = torch.optim.Adam([self.weight], lr=learning_rate)
        for t in range(n_epochs):

            jointLoss=0
            if (counter+1)*batchSize >= sig_obs_pairs.shape[0]:
                counter=0
                sig_obs_pairs=fastShuffle(sig_obs_pairs,1)

            batch_vectors = sig_obs_pairs[counter*batchSize:(counter+1)*batchSize, :]
            observations = batch_vectors[:,1].astype(np.float32)
            signals = batch_vectors[:,0].astype(np.float32)
            observations = torch.from_numpy(observations.astype(np.float32)).float().to(self.device)
            signals = torch.from_numpy(signals).float().to(self.device)
            p = self.likelihood(observations, signals)
            loss=torch.mean(-torch.log(p))
            jointLoss=jointLoss+loss
            
            if t%100==0:
                print(t, jointLoss.item())
                

            if (t%(int(n_epochs*0.5))==0):
                trained_weight = self.weight.cpu().detach().numpy()
                min_signal = self.min_signal.cpu().detach().numpy()
                max_signal = self.max_signal.cpu().detach().numpy()
                np.savez(self.path+name, trained_weight=trained_weight, min_signal = min_signal, max_signal = max_signal, min_sigma = self.min_sigma)

                
            

            optimizer.zero_grad()
            jointLoss.backward()
            optimizer.step()
            counter+=1

        print("===================\n")    
        print("The trained parameters (" + name + ") is saved at location: "+ self.path)


