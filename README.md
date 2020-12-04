## Efficient semidefinite-programming-based inference for binary and multi-class MRFs

This repository contains code for the NeurIPS 2020 paper [Efficient semidefinite-programming-based inference for binary and multi-class MRFs](https://proceedings.neurips.cc/paper/2020/file/2cb274e6ce940f47beb8011d8ecb1462-Paper.pdf) by Chirag Pabbaraju, [Po-Wei Wang](https://powei.tw/) and [J. Zico Kolter](http://zicokolter.com/). 

All the code was run on a MacBook Pro with the following specs - Processor: 2.3GHz Dual-Core Intel Core i5,
    Memory: 16GB, Graphics: Intel Iris Plus Graphics 640 1536 MB

## Installation

### Via pip
```bash
pip install sdp-mrf
```

### Via source
Refer to the initial setup code in [this Collab notebook](https://colab.research.google.com/drive/1df6Y1XsYdzNBwHAqAC_3IggeNYagKgKR?usp=sharing)

## Example usage
Instantiate a Potts model on a 4-class Potts model on 10 variables, and compute the MAP estimate and partition function for it.
```
n, k = 10, 4
A, h = np.random.rand(n, n), np.random.rand(n, k)
A = (A + A.T) / 2
p = PottsModel(A, h, k)
mode_x, mode_f = model.solve_map(solver='M4')
logZ = model.solve_partition_function(solver='M4')
```

## Experiments

We provide data for all the experiments done in the paper in the data/ directory. Run ``` sh download_pydensecrf_spectral_ai.sh``` to download pydensecrf (https://github.com/lucasb-eyer/pydensecrf) and spectral AI (https://github.com/sejunpark-repository/spectral_approximate_inference). We have only added a new method vectorInference() in the DenseCRF class in pydensecrf in order to support mixing method updates in our formulation. The script automatically applies the patch (vectorInf.diff) and adds this method. For Spectral AI, we only add a new file (check.m) in the matlab code in order to run their algorithm on ER graphs.

We also provide documented iPython notebooks for each experiment performed in the paper in the notebooks/ directory.
+ Data Generation Binary MRFs.ipynb: Synthetic data generation for binary MRFs according to the procedure described
in the experiments section in the paper. Note that for complete binary graphs, Park et al. already provide the dataset
as part of their code.
+ Data Generation Multi-class MRFs.ipynb: Synthetic data generation for multi-class MRFs according to the procedure
described in the paper.
+ logZ estimation - Binary MRFs.ipynb: Code for comparisons between Wang et al. (ICLR 2014), Park et al. (ICML 2019), AIS
and our method for computing log-partition function in binary (complete and ER) MRFs.
+ logZ estimation - Multi-class MRFs.ipynb: Code for comparisons between AIS and our method for computing the log-partition
function in k-class complete MRFs (for k=3,4,5).
+ MAP Estimation.ipynb: Code for mode estimation in k-class complete MRFs (k=3,4,5). For 100 synthetic MRFs with coupling
stregth 2.5, we run both AIS and our method and keep track of the present estimate of mode with each iteration of the
sampling.
+ Image Segmentation.ipynb: Code for the image segmentation tasks on two images (swan and bench) provided in the data/
folder. The code makes use of the high-dimensional filtering method provided by pydensecrf (we have modified it and
included a method called vectorInf() specifically for our needs).
