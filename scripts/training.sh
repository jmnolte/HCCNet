#!/bin/bash
# parameters for slurm
#SBATCH -c 32                                   # number of cores, 32
#SBATCH --job-name=milmodel                     # job name
#SBATCH -e milmodel_smooth.err                         # error file
#SBATCH --gres=gpu:1                            # number of gpus 4
#SBATCH --mem=120gb                             # Job memory request
#SBATCH -o milmodel_smooth.out                         # output file
 
# show actual node in output file, usefull for diagnostics
hostname
 
# load all required software modules
module load nvidia/cuda-11.6
module load miniconda3/3.8
# activate base environment
source activate base
# activate hcc-gpu environment
conda activate hcc-gpu  # activate my conda environment
 
# It's nice to have some information logged for debugging
echo "Gpu devices                 : "$CUDA_VISIBLE_DEVICES
echo "Starting worker: "

# defaults: prob = 0.5; total batch size = 16; lr = 1e-6 (cyclic); lr = 1e-4 (exp); lw_finetune = 13; weight deacy = 1e-2
# no difference between augmentation probabilities (0.25 to 0.5) or between weight decays (1e-4 to 1e-6)
torchrun \
    --standalone \
    --nproc_per_node=1 /home/x3007104/thesis/scripts/training.py \
    --num-classes 2 \
    --mil-mode att \
    --backbone resnet50 \
    --tl-strategy lw_finetune \
    --cutoff-point 13 \
    --smoothing-coef 0.05 \
    --pretrained \
    --distributed \
    --amp \
    --early-stopping \
    --augment-prob 0.5 \
    --train-ratio 0.75 \
    --num-patches 32 \
    --batch-size 4 \
    --total-batch-size 16 \
    --learning-rate 1e-6 \
    --weight-decay 1e-2 \
    --num-workers 8 \
    --mod-list DWI_b150 \
    --seed 1234 \
    --data-dir /deepstore/datasets/bms/hcc_study \
    --results-dir /home/x3007104/thesis/results