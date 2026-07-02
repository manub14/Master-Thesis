# Pre- and Post-Encoding Augmentations in Spiking Neural Networks

This repository contains the code for the master's thesis *Pre- and
Post-Encoding Augmentations in Spiking Neural Networks*, carried out at
Universität Hildesheim.

The work studies data augmentation for spiking neural networks (SNNs) at two
different stages of the pipeline. Pre-encoding augmentations act on the input
image before the spiking patch embedding. Post-encoding augmentations act on
the spike feature tensor of shape (T, B, C, H, W) after the spiking patch
embedding. The main architecture is the Spike-Driven Transformer (SDT), and
MS-ResNet is used as a second spiking backbone.

## Abstract

> **TODO: replace this with the abstract from your thesis.**

Data augmentation improves generalization and robustness in deep learning, but
most existing methods are designed for artificial neural networks and do not
account for the temporal, spike-based representation used by spiking neural
networks. This thesis studies augmentation for SNNs at two stages of the
pipeline. Pre-encoding methods operate on the input image before spiking patch
embedding, while post-encoding methods operate directly on the spike feature
tensor after embedding. We implement and evaluate methods from both families on
the Spike-Driven Transformer and MS-ResNet backbones, across static and
neuromorphic datasets, and we report accuracy, robustness to corruptions, and
estimated energy cost.

## Methods

Pre-encoding (applied to the input image before spiking patch embedding):

- PatchShuffle
- PatchDropout
- PatchMix
- IPMix
- LayerMix

Post-encoding (applied to the spike feature tensor of shape (T, B, C, H, W)
after spiking patch embedding):

- TimeShuffle
- TimeMask
- TimeMix
- LocalTimeShuffle
- TemporalJitter
- PAPMix (FullDimMix)
- CenterPatchMinLift
- Frequency Encoding
- PatchShuffle Postenc
- PatchDropout Postenc
- PatchMix Postenc

## Repository Structure

- `Spike-Driven-Transformer/`: main codebase with the full set of augmentations
  on the SDT backbone.
- `MS-ResNet/`: the same study applied to the MS-ResNet spiking backbone.
- `Spike-Driven-Transformer-timeshuffle/`: experiment variant. *TODO: one line
  on what this run isolates.*
- `Spike-Driven-Transformer-sameclass/`: experiment variant. *TODO: one line.*
- `Spike-Driven-Transformer-neighbouring0sto1s/`: experiment variant. *TODO:
  one line.*
- `Spike-Driven-Transformer-rsm/`: experiment variant. *TODO: one line.*

Each folder shares a common layout. `model/` and `module/` hold the network
definition, `conf/` holds configuration files, `dvs_utils/` holds helpers for
the event-based datasets, `train.py` is the training entry point, `criterion.py`
and `data.py` define the loss and the data loading, and `firing_num.py` and
`plot_spike_ratio.py` handle spike-rate analysis.

## Datasets

Evaluated on CIFAR-10,CIFAR-10-C, CIFAR-100,CIFAR-100-C, Tiny ImageNet, CIFAR10-DVS, and DVS128Gesture.

> **TODO:** state where to download each dataset and where to place it, for
> example a `data/` folder that is ignored by git.

## Setup

```
git clone https://github.com/manub14/Master-Thesis.git
cd Master-Thesis/Spike-Driven-Transformer

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install PyTorch for your own CUDA or CPU setup by following the instructions at
https://pytorch.org/get-started/locally/. The spiking layers use spikingjelly.

## Running

From inside one of the code folders:

```
python train.py
```

or, if a shell script is present:

```
bash train.sh
```

> **TODO:** add one line describing the main arguments of `train.py`, for
> example how to select the dataset and which augmentation to apply.

## Results

> **TODO:** add a short summary of the main results, or point to the result
> tables in the thesis.

## Citation

> **TODO:** add the citation for your thesis once it is finalized.

## Acknowledgements

- The Spike-Driven Transformer backbone builds on the work of Yao et al.
  *TODO: add a link to the original repository.*
- The MS-ResNet backbone builds on the work of Hu et al. *TODO: add a link.*
- Spiking neuron models are provided by the spikingjelly library.
