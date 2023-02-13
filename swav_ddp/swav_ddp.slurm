#!/bin/bash

## change the last two digits to your team id
#SBATCH --account=csci_ga_2572_2022sp_01

## change the partition number to use different number of GPUs
##SBATCH --partition=n1s8-v100-1
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=8

##SBATCH --partition=n1s16-v100-2
##SBATCH --gres=gpu:2
##SBATCH --tasks-per-node=2
##SBATCH --cpus-per-task=8


#SBATCH --partition=n1c24m128-v100-4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24


#SBATCH --time=24:00:00
#SBATCH --output=swavddp_%j.out
#SBATCH --error=swavddp_%j.err
#SBATCH --exclusive
#SBATCH --requeue

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

cp -rp /scratch/DL22SP/unlabeled_224.sqsh /tmp
cp -rp /scratch/DL22SP/labeled.sqsh /tmp
echo "Dataset is copied to /tmp"
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES

##export MASTER_ADDR=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
##export MASTER_PORT=50000

##master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
##dist_url="tcp://"
##dist_url+=$master_node
##dist_url+=:50000

singularity exec --nv \
--bind /scratch \
--overlay /scratch/DL22SP/conda.ext3:ro \
--overlay /tmp/unlabeled_224.sqsh \
--overlay /tmp/labeled.sqsh \
/share/apps/images/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate
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
"
