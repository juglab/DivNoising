import torch
import numpy as np
import time
from sklearn.feature_extraction import image
from tqdm import tqdm
from glob import glob
from sklearn.cluster import MeanShift
from matplotlib import pyplot as plt
from IPython.display import clear_output

def normalize(img, mean, std):
    """Normalize an array of images with mean and standard deviation. 
        Parameters
        ----------
        img: array
            An array of images.
        mean: float
            Mean of img array.
        std: float
            Standard deviation of img array.
        """
    return (img - mean)/std

def denormalize(img, mean, std):
    """Denormalize an array of images with mean and standard deviation. 
    Parameters
    ----------
    img: array
        An array of images.
    mean: float
        Mean of img array.
    std: float
        Standard deviation of img array.
    """
    return (img * std) + mean

def convertToFloat32(train_images,val_images):
    """Converts the data to float 32 bit type. 
    Parameters
    ----------
    train_images: array
        Training data.
    val_images: array
        Validation data.
        """
    x_train = train_images.astype('float32')
    x_val = val_images.astype('float32')
    return x_train, x_val

def getMeanStdData(train_images,val_images):
    """Compute mean and standrad deviation of data. 
    Parameters
    ----------
    train_images: array
        Training data.
    val_images: array
        Validation data.
    """
    x_train_ = train_images.astype('float32')
    x_val_ = val_images.astype('float32')
    data = np.concatenate((x_train_,x_val_), axis=0)
    mean, std = np.mean(data), np.std(data)
    return mean, std

def convertNumpyToTensor(numpy_array):
    """Convert numpy array to PyTorch tensor. 
    Parameters
    ----------
    numpy_array: numpy array
        Numpy array.
    """
    return torch.from_numpy(numpy_array)

def extract_patches(x,patch_size,num_patches):
    """Deterministically extract patches from array of images. 
    Parameters
    ----------
    x: numpy array
        Array of images.
    patch_size: int
        Size of patches to be extracted from each image.
    num_patches: int
        Number of patches to be extracted from each image.    
    """
    patches = np.zeros(shape=(x.shape[0]*num_patches,patch_size,patch_size))
    
    for i in tqdm(range(x.shape[0])):
        patches[i*num_patches:(i+1)*num_patches] = image.extract_patches_2d(x[i],(patch_size,patch_size), num_patches,
                                                                           random_state=i)    
    return patches

def augment_data(X_train):
    """Augment data by 8-fold with 90 degree rotations and flips. 
    Parameters
    ----------
    X_train: numpy array
        Array of training images.
    """
    X_ = X_train.copy()

    X_train_aug = np.concatenate((X_train, np.rot90(X_, 1, (1, 2))))
    X_train_aug = np.concatenate((X_train_aug, np.rot90(X_, 2, (1, 2))))
    X_train_aug = np.concatenate((X_train_aug, np.rot90(X_, 3, (1, 2))))
    X_train_aug = np.concatenate((X_train_aug, np.flip(X_train_aug, axis=1)))

    print('Raw image size after augmentation', X_train_aug.shape)
    return X_train_aug

def loadImages(path):
    """Load images from a given directory. 
    Parameters
    ----------
    path: String
        Path of directory from where to load images from.
    """
    files = sorted(glob(path))
    data=[]
    print(path)
    for f in files:
        if '.png' in f:
            im_b = np.array(io.imread(f))
        if '.npy' in f:
            im_b = np.load(f)
        data.append(im_b)
        
    data = np.array(data).astype(np.float32)
    return data

def getSamples(vae, size=20,zSize=64, mu=None, logvar=None, samples=1, tq=False):    
    """Generate synthetic samples from DivNoising network. 
    Parameters
    ----------
    vae: VAE Object
        DivNoising model.
    size: int
        Size of generated image in the bottleneck.
    zSize: int
        Dimension of latent space for each pixel in bottleneck.
    mu: PyTorch tensor
        latent space mean tensor.
    logvar: PyTorch tensor
        latent space log variance tensor.
    samples: int
        Number of synthetic samples to generate.
    tq: boolean
        If tqdm should be active or not to indicate progress.
    """
    if mu is None:
        mu=torch.zeros(1,zSize,size,size).cuda()
    if logvar is None:    
        logvar=torch.zeros(1,zSize,size,size).cuda()

    results=[]
    for i in tqdm(range(samples),disable= not tq):
        z = vae.reparameterize(mu, logvar)
        recon = vae.decode(z)
        recon_cpu = recon.cpu()
        recon_numpy = recon_cpu.detach().numpy()
        recon_numpy.shape=(recon_numpy.shape[-2],recon_numpy.shape[-1]) 
        results.append(recon_numpy) 
    return np.array(results)

def interpolate(vae, z_start, z_end, steps, display, vmin=0,vmax=255,):
    results=[]
    for i in range(steps):
        alpha=(i/(steps-1.0))
        z=z_end*alpha + z_start*(1.0-alpha)
        recon = vae.decode(z)
        recon_cpu = recon.cpu()
        recon_numpy = recon_cpu.detach().numpy()
        recon_numpy.shape=(recon_numpy.shape[-2],recon_numpy.shape[-1])
        if display:
            clear_output(wait=True)
            plt.imshow(recon_numpy,vmin=vmin, vmax=vmax)
            plt.show()
        time.sleep(0.4)
        results.append(recon_numpy)
    return results

def tiledMode(im, ps, overlap, display=True, vmin=0,vmax=255,
              initBW=200, minBW=100, reduce=0.9):
    means=np.zeros(im.shape[1:])
    xmin=0
    ymin=0
    xmax=ps
    ymax=ps
    ovLeft=0
    while (xmin<im.shape[2]):
        ovTop=0
        while (ymin<im.shape[1]):

            inputPatch=im[:,ymin:ymax,xmin:xmax]
            a = findMode(inputPatch,
                        initBW, minBW, reduce)
            a=a[:a.shape[0], :a.shape[1]] 
            means[ymin:ymax,xmin:xmax][ovTop:,ovLeft:] = a[ovTop:,ovLeft:]

            ymin=ymin-overlap+ps
            ymax=ymin+ps
            ovTop=overlap//2
        
        ymin=0
        ymax=ps
        xmin=xmin-overlap+ps
        xmax=xmin+ps
        ovLeft=overlap//2
        
        if display:
            plt.imshow(means,vmin=vmin, vmax=vmax)
            plt.show()
            clear_output(wait=True)
        
        
    return means


def findClosest(samples, q):
    """Find closest sample to a given sample. 
    Parameters
    ----------
    samples: array
        Array of samples from which the closest image needs to be found.
    q: image(array)
        Image to which the closest image needs to be found.
    """
    dif=np.mean(np.mean((samples-q)**2, -1),-1)
    return samples[np.argmin(dif)]

def findMode(samples, initBW=200, minBW=100, reduce=0.9):
    """Find the modes of a distribution of images. 
    Parameters
    ----------
    samples: array
        Array of samples from which the modes need to be found.
    initBW: int
        Initial bandwidth.
    minBW: int
        Minimum bandwidth.
    reduce: float
        Factor by which to reduce bandwith i n iterations.
    """
    imagesC=samples.copy()
    imagesC.shape=(samples.shape[0],samples.shape[1]*samples.shape[2])
    seed=np.mean(imagesC,axis=0)[np.newaxis,...]
    bw=initBW
    for i in range(15):

        clustering = MeanShift(bandwidth=bw, seeds=seed, cluster_all=True).fit(imagesC)
        centers=clustering.cluster_centers_.copy()
        seed=centers 
        bw=bw*reduce
        if bw < minBW: 
            break
            
    result=seed[0]
    result.shape=(samples.shape[1],samples.shape[2]) 
    return result

def plotProbabilityDistribution(signalBinIndex, histogram, gaussianMixtureNoiseModel, min_signal, max_signal, n_bin, device):
    """Plots probability distribution P(x|s) for a certain ground truth signal. 
       Predictions from both Histogram and GMM-based Noise models are displayed for comparison.
        Parameters
        ----------
        signalBinIndex: int
            index of signal bin. Values go from 0 to number of bins (`n_bin`).
        histogram: numpy array
            A square numpy array of size `nbin` times `n_bin`.
        gaussianMixtureNoiseModel: GaussianMixtureNoiseModel
            Object containing trained parameters.
        min_signal: float
            Lowest pixel intensity present in the actual sample which needs to be denoised.
        max_signal: float
            Highest pixel intensity present in the actual sample which needs to be denoised.
        n_bin: int
            Number of Bins.
        device: GPU device
        """
    histBinSize=(max_signal-min_signal)/n_bin
    querySignal_numpy= (signalBinIndex/float(n_bin)*(max_signal-min_signal)+min_signal)
    querySignal_numpy +=histBinSize/2
    querySignal_torch = torch.from_numpy(np.array(querySignal_numpy)).float().to(device)
    
    queryObservations_numpy=np.arange(min_signal, max_signal, histBinSize)
    queryObservations_numpy+=histBinSize/2
    queryObservations = torch.from_numpy(queryObservations_numpy).float().to(device)
    pTorch=gaussianMixtureNoiseModel.likelihood(queryObservations, querySignal_torch)
    pNumpy=pTorch.cpu().detach().numpy()
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.xlabel('Observation Bin')
    plt.ylabel('Signal Bin')
    plt.imshow(histogram**0.25, cmap='gray')
    plt.axhline(y=signalBinIndex+0.5, linewidth=5, color='blue', alpha=0.5)
    
    plt.subplot(1, 2, 2)
    plt.plot(queryObservations_numpy, histogram[signalBinIndex, :]/histBinSize, label='GT Hist: bin ='+str(signalBinIndex), color='blue', linewidth=2)
    plt.plot(queryObservations_numpy, pNumpy, label='GMM : '+' signal = '+str(np.round(querySignal_numpy,2)), color='red',linewidth=2)
    plt.xlabel('Observations (x) for signal s = ' + str(querySignal_numpy))
    plt.ylabel('Probability Density')
    plt.title("Probability Distribution P(x|s) at signal =" + str(querySignal_numpy))
    plt.legend()
    
def predictMMSE(vae, img, samples, returnSamples=False, tq=True): 
    '''
    Predicts MMSE estimate.
    Parameters
    ----------
    vae: VAE object
        DivNoising model.
    img: array
        Image for which denoised MMSE estimate needs to be computed.
    samples: int
        Number of samples to average for computing MMSE estimate.
    returnSamples: 
        Should the method also return the individual samples?
    tq: boolean
        If tqdm should be active or not to indicate progress.
    '''
    img_height,img_width=img.shape[0],img.shape[1]
    img_t = torch.Tensor(img)
    image_sample = img_t.view(1,1,img.shape[0], img.shape[1]).cuda()
    mu, logvar = vae.encode(image_sample)
    akku=np.zeros((img_height,img_width))
    
    if returnSamples:
        retSamp=[]
    for i in tqdm(range(samples), disable= not tq):
        z = vae.reparameterize(mu, logvar)
        #z=mu
        recon = vae.decode(z)
        recon_cpu = recon.cpu()
        recon_numpy = recon_cpu.detach().numpy()
        recon_numpy.shape=(img_height,img_width) 
        akku+=recon_numpy
        if returnSamples:
            retSamp.append(recon_numpy)
        
    akku=akku/float(samples)
    if returnSamples:
        return akku, np.array(retSamp)
    else:
        return akku

def PSNR(gt, img):
    '''
    Compute PSNR.
    Parameters
    ----------
    gt: array
        Ground truth image.
    img: array
        Predicted image.''
    '''
    mse = np.mean(np.square(gt - img))
    return 20 * np.log10(np.max(gt)-np.min(gt)) - 10 * np.log10(mse)