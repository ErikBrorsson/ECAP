# ECAP: Extensive Cut-and-Paste Augmentation for Unsupervised Domain Adaptive Semantic Segmentation

## Abstract
We consider unsupervised domain adaptation (UDA) for semantic
segmentation in which the model is trained on a labeled source
dataset and adapted to an unlabeled target dataset. Unfortu-
nately, current self-training methods are susceptible to misclassified
pseudo-labels resulting from erroneous predictions. Since certain
classes are typically associated with less reliable predictions in
UDA, reducing the impact of such pseudo-labels without skewing
the training towards some classes is notoriously difficult. To this end,
we propose an extensive cut-and-paste strategy (ECAP) to leverage
reliable pseudo-labels through data augmentation. Specifically,
ECAP maintains a memory bank of pseudo-labeled target samples
throughout training and cut-and-pastes the most confident ones onto
the current training batch. We implement ECAP on top of the recent
method MIC and boost its performance on two synthetic-to-real do-
main adaptation benchmarks. Notably, MIC+ECAP reaches an un-
precedented performance of 69.1 mIoU on the Synthia $\rightarrow$ Cityscapes
benchmark.

## Results
To come.
## Checkpoints
To come.

# Instructions
ECAP is implemented on top of the code base of HRDA and MIC.
The source code for each of these projects, with the addition of ECAP, are contained in the respective folders. Since we make minimal adjustments to these repositories, we refer the reader to the corresponding subdirectories for elaborate instruction on using the code.
## Training
In each of the directories hrda and mic, we include the relevant configuration files for ECAP to reproduce the results in our paper. For example, to train MIC+ECAP on GTA $\rightarrow$ Cityscapes, simply navigate to the mic directory and run the following command.

python run_experiments.py --config configs/ecap/table1_sota/mic_ecap_gta0.py

## Testing
After downloading, for example, the checkpoint for HRDA+ECAP on GTA and extracting the .tar.gz file under hrda/work_dirs, the model can be tested by running the following command.

sh test.sh work_dirs/hrda_ecap_gta


# Ackowledgement
We thank the authors of the following repositories for making their code publicly available.

* [MIC](https://github.com/lhoyer/MIC)
* [HRDA](https://github.com/lhoyer/HRDA)
* [DAFormer](https://github.com/lhoyer/DAFormer)
* [DACS](https://github.com/vikolss/DACS)
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [SegFormer](https://github.com/NVlabs/SegFormer)
