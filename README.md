# CAPE-ML4Sci

This repository contains the code for the paper:
CAPE: Channel-Attention-Based PDE Parameter Embeddings for SciML

In this work, we provide a code to reproduce the above paper experiments. 
This repository consists of the codes training and evaluating different machine learning models as baseline.

## Dataset

We made use of the dataset provided by PDEBench: 
https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986

## Installation

```bash
pip install --upgrade pip wheel
pip install -r requirements.txt
```

## Requirements
In our paper, we used the following GPUs:  
 GeForce RTX 2080 GPU for 1D PDEs
 GeForce GTX 3090 for 2D

## Baseline Models
In this work, we provide two different ML models with CAPE module to be trained and evaluated against the benchmark datasets, namely [FNO](https://arxiv.org/pdf/2010.08895.pdf) and [U-Net](https://www.sciencedirect.com/science/article/abs/pii/S0010482519301520?via%3Dihub).
The examples of the training scripts for the baseline model are contained in:
- `run_training_PrmEmb_Adv.sh` is the main script to train and evaluate the FNO for 1D Advection equation. 
- `run_training_PrmEmb_Adv_Unet.sh` is the main script to train and evaluate the Unet for 1D Advection equation. - 
- `run_training_PrmEmb_Bgs.sh` is the main script to train and evaluate the FNO for 1D Burgers equation. 
- `run_training_PrmEmb_Bgs_Unet.sh` is the main script to train and evaluate the Unet for 1D Burgers equation. - 
- `run_training_PrmEmb_2DCFD.sh` is the main script to train and evaluate the FNO for 2D NS equations. 
- `run_training_PrmEmb_2DCFD_Unet.sh` is the main script to train and evaluate the Unet for 2D NS equations. - 

```bash
sh run_training_PrmEmb_Adv.sh
```
(Note that training data should be downloaded from the PDEBench data server.)

## Code contributors

* Deleted for purposes of Anonymity

## License 
MIT for solver code and baseline code, and Anonymised License for selected code 
