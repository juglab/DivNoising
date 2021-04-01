# DivNoising: Diversity Denoising with Fully Convolutional Variational Autoencoders

Mangal Prakash<sup>1</sup>, Alexander Krull<sup>1,2</sup>, Florian Jug<sup>2</sup></br>
<sup>1</sup>Authors contributed equally, <sup>2</sup>Shared last authors. <br>
Max Planck Institute of Molecular Cell Biology and Genetics (**[MPI-CBG](https://www.mpi-cbg.de/home/)**) <br>
Center for Systems Biology (**[CSBD](https://www.csbdresden.de/)**) in Dresden, Germany .

![teaserFigure]( https://github.com/juglab/DivNoising_RC/blob/Readme/resources/Fig2_full.png "Figure 1 taken from publication")

Deep Learning based methods have emerged as the indisputable leaders for virtually all image restoration tasks. Especially in the domain of microscopy images, various content-aware image restoration (CARE) approaches are now used to improve the interpretability of acquired data. But there are limitations to what can be restored in corrupted images, and any given method needs to make a sensible compromise between many possible clean signals when predicting a restored image. Here, we propose DivNoising - a denoising approach based on fully-convolutional variational autoencoders, overcoming this problem by predicting a whole distribution of denoised images. Our method is unsupervised, requiring only noisy images and a description of the imaging noise, which can be measured or bootstrapped from noisy data. If desired, consensus predictions can be inferred from a set of DivNoising predictions, leading to competitive results with other unsupervised methods and, on occasion, even with the supervised state-of-the-art. DivNoising samples from the posterior enable a plethora of useful applications. We are (i) discussing how optical character recognition (OCR) applications could benefit from diverse predictions on ambiguous data, and (ii) show in detail how instance cell segmentation gains performance when using diverse DivNoising predictions.

### Information

This repository hosts the the code for the **[publication](https://openreview.net/pdf?id=agHLCOBM5jP)** **Fully Unsupervised Diversity Denoising with Convolutional Variational Autoencoders**. 

### Citation
If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{
prakash2021fully,
title={Fully Unsupervised Diversity Denoising with Convolutional Variational Autoencoders},
author={Mangal Prakash and Alexander Krull and Florian Jug},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=agHLCOBM5jP}
}
```

### Dependencies 
We have tested this implementation using pytorch version 1.1.0 and cudatoolkit version 9.0.

In order to replicate results mentioned in the publication, one could use the same virtual environment (`DivNoising.yml`) as used by us. Create a new environment, for example, by entering the python command in the terminal `conda env create -f path/to/DivNoising.yml`.


### Getting Started
Look in the `examples` directory and try out the notebooks. Prior to beginning the denoising pipeline, prepare the noise model in case your noisy data is NOT generated by a gaussian noise process. For such cases, begin by running : (i) `Convallaria-CreateNoiseModel.ipynb`. This will download the data and create a suitable noise model. (ii) `Convallaria-Training.ipynb`. This starts network training. (iii) `Convallaria-Prediction.ipynb`. This starts prediction part.
DivNoising is based on Variational Autoencoders, hence, it is a generative model. You can use it to generate synthetic clean images not present in training/validation/test set. This can be achieved by running the notebook (iv) `Convallaria-Image_Generation.ipynb`.

In case, your noisy data is generated using a Gaussian noise model, then you can start with the training step directly.

