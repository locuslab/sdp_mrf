## Efficient semidefinite-programming-based inference for binary and multi-class MRFs

This repository contains code for the paper "Efficient semidefinite-programming-based inference for binary and multi-class MRFs" published at NeurIPS 2020. We have provided documented iPython notebooks for each experiment performed in the paper.

All the code was run on a MacBook Pro with the following specs - Processor: 2.3GHz Dual-Core Intel Core i5,
    Memory: 16GB, Graphics: Intel Iris Plus Graphics 640 1536 MB

Organization:

1) data/ - has all the data for all the experiments done in the paper.

2) download_pydensecrf_spectral_ai.sh - Run this script to download pydensecrf (https://github.com/lucasb-eyer/pydensecrf) and spectral AI
    (https://github.com/sejunpark-repository/spectral_approximate_inference). We have only added
    a new method vectorInference() in the DenseCRF class in pydensecrf in order to support mixing method updates in our formulation. The script automatically 
    applies the patch (vectorInf.diff) and adds this method. For Spectral AI, we only add a new file (check.m) in the matlab code in order to run their algorithm 
    on ER graphs.

3) notebooks/ - has all the iPython Notebooks
    + Data Generation Binary MRFs.ipynb: Synthetic data generation for binary MRFs according to the procedure described
    in the experiments section in the paper. Note that for complete binary graphs, Park et al. already provide the dataset
    as part of their code.
    + Data Generation Multi-class MRFs.ipynb: Synthetic data generation for multi-class MRFs according to the procedure
    described in the paper.
    + logZ estimation - Binary MRFs.ipynb: Code for comparisons between Wang et al. (ICLR 2014), Park et al. (ICML 2019), AIS
    and our method for computing log-partition function in binary (complete and ER) MRFs.
    + logZ estimation - Multi-class MRFs.ipynb: Code for comparisons between AIS and our method for computing the log-partition
    function in k-class complete MRFs (for k=3,4,5).
    + Modes Binary MRFs.ipynb: Code for mode estimation in binary (complete and ER) MRFs. For 100 synthetic MRFs with coupling
    stregth 2.5, we run both AIS and our method and keep track of the present estimate of mode with each iteration of the
    sampling.
    + Modes Multi-class MRFs.ipynb: Code for mode estimation in k-class complete MRFs (k=3,4,5).
    + Image Segmentation.ipynb: Code for the image segmentation tasks on two images (swan and bench) provided in the data/
    folder. The code makes use of the high-dimensional filtering method provided by pydensecrf (we have modified it and
    included a method called vectorInf() specifically for our needs).
    + part.c, part.h - These files have source code for the mixing method written in a slightly more optimized manner.
