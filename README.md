# ECAP: Extensive Cut-and-Paste Augmentation for Unsupervised Domain Adaptive Semantic Segmentation

## Abstract
We consider unsupervised domain adaptation (UDA) for semantic
segmentation in which the model is trained on a labeled source
dataset and adapted to an unlabeled target dataset. Unfortunately, current self-training methods are susceptible to misclassified
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
The two tables below show the main results presented in the paper, which demonstrate the efficacy of our method on synthetic-to-real UDA.
<img src=resources/sota_comparison.png width=1200>
<img src=resources/complementary_results.png width=600>

## Checkpoints
* [MIC+ECAP GTA->Cityscapes](https://drive.google.com/file/d/1IFWE8zKBpsOI37NQ8tYLK9LMnvjlGKPl/view?usp=sharing)

* [MIC+ECAP Synthia->Cityscapes](https://drive.google.com/file/d/1upQhfPPtdNpjPr6tmfXEAiIRAF50iZx-/view?usp=sharing)

* [MIC+ECAP Cityscapes->DarkZurich](https://drive.google.com/file/d/1Yxzx432Lt97mwxo1vxgzzSQOKVTbHC31/view?usp=drive_link)

* [MIC+ECAP Cityscapes->ACDC](https://drive.google.com/file/d/1nAKQu7uxJ3o2QdoIBh0UVaVeGl_32GxH/view?usp=drive_link)

# Instructions
ECAP is implemented on top of the code base of HRDA and MIC.
The source code for each of these projects, with the addition of ECAP, are contained in the respective folders. Since we make minimal adjustments to these repositories, we refer the reader to the corresponding subdirectories for elaborate instruction on using the code.
## Training
In each of the directories hrda and mic, we include the relevant configuration files for ECAP to reproduce the results in our paper. For example, to train MIC+ECAP on GTA $\rightarrow$ Cityscapes, simply navigate to the mic directory and run the following command.

python run_experiments.py --config configs/ecap/table1_sota/mic_ecap_gta0.py

## Testing
After downloading, for example, the checkpoint MIC+ECAP GTA->Cityscapes and extracting the .tar.gz file under mic/work_dirs, the model can be tested by running the following command.

sh test.sh work_dirs/*targz_filename*


# Ackowledgement
We thank the authors of the following repositories for making their code publicly available.

* [MIC](https://github.com/lhoyer/MIC)
* [HRDA](https://github.com/lhoyer/HRDA)
* [DAFormer](https://github.com/lhoyer/DAFormer)
* [DACS](https://github.com/vikolss/DACS)
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [SegFormer](https://github.com/NVlabs/SegFormer)
