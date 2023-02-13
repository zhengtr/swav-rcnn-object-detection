# CSCI-GA 2572 (Deep Learning, Spring 2022) Final Project.
## Courant Institute of Mathematical Sciences - New York University
## Self-supervised learning for object detection

Created by *Tanran Zheng*, *Guojin Tang* and *Deep Mehta*.

# Setup
Two parts: (1) train a backbone with *SwAV* model using unlabeled data, and (2) train Faster RCNN using the pretrained backbone using labeled data. 

## Part 1
Train SwAV

**Data**  
Unlabled Data

### Module:  swav_ddp
Train swav with multiple GPU on single node  

**Scripts**  
`swav_ddp.py`: contains the main function to run the implementation  

**To run**  
To reproduce our *ResNet50* backbone:

Run
```
python -u swav_ddp.py \
--workers 2 \
--epochs 100 \
--batch_size 32 \
--epsilon 0.05 \
--base_lr 0.6 \
--final_lr 0.0006 \
--warmup_epochs 0 \
--size_crops 224 96 \
--nmb_crops 2 6 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--use_fp16 true \
--freeze_prototypes_niters 5005 \
--queue_length 3072 \
--nmb_prototypes 2000 \
--epoch_queue_starts 30 \
--sync_bn pytorch \
--syncbn_process_group_size 4 \
--world_size 1 \
--rank 0 \
--arch resnet50
```
or run

`swav_ddp.slurm`: slurm file for NYU hpc.

### Module:  swav_simplified
Train swav with single GPU on single node  

**Scripts**  
`main_swav.py`: contains the main function to run the implementation  
`resnet50.py`: modified file contains impl of ResNet architecture 

**To run**  
To reproduce our *ResNet18* backbone:

Run
```
python -u main_swav.py \
--epochs 100 \
--queue_length 3840 \
--batch_size 128 \
--base_lr 0.01 \
--final_lr 0.00001 \
--warmup_epochs 10 \
--start_warmup 0.3 \
--arch resnet18 \
--freeze_prototypes_niters 5000 \
--checkpoint_freq 3 \
--data_path ../unlabeled_data \
--nmb_prototypes 1000\
```

## Part 2
Train Faster RCNN

**Data**  
labled Data

**Scripts**  
`train_submission.py`: Load backbone and train Faster RCNN with customized module. *USED FOR FINAL SUBMISSION*  
`train.py`: A more generalized version, contains more options of the architection in order to conduct experiments. No major difference from `train_submission.py`, only with more args.  
`swav_resnet.py`: ResNet architecture of the backbone  
`cusomized_module.py`: Customized module of RCNN, including two different Box Head, FPN, and FrozenBatchNorm2d.  
`build_model.py`: Impl to replace specific layers in backbone  
`utils.py`: Additional impl for logging and checkpoint resume included comparing to original demo code.

**To run**  
To reproduce our final *rcnn* model:

Run
```
python train_submission.py \
--epochs 30 \
--data_path /labeled \
--gcp_sucks 1 \
--eval_freq 2 \
--checkpoint_freq 2 \
--sched_step 6 \
--swav_file <backbone.pth> \
--sched_gamma 0.1
```
or run

`train_rcnn.slurm`: slurm file for NYU hpc.

## Others
`normalization.py` and `normalization_labeled.py`: script to calculate the mean and variance of the dataset.

# Major reference

### SwAV: 
https://github.com/facebookresearch/swav
```
@article{DBLP:journals/corr/abs-2006-09882,
  author    = {Mathilde Caron and
               Ishan Misra and
               Julien Mairal and
               Priya Goyal and
               Piotr Bojanowski and
               Armand Joulin},
  title     = {Unsupervised Learning of Visual Features by Contrasting Cluster Assignments},
  journal   = {CoRR},
  volume    = {abs/2006.09882},
  year      = {2020},
  url       = {https://arxiv.org/abs/2006.09882},
  eprinttype = {arXiv},
}
```
### MOCO: 
https://github.com/facebookresearch/moco
