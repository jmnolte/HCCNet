#!/bin/bash
# parameters for slurm
#SBATCH -c 32                                   # number of cores, 32
#SBATCH --job-name=inference                    # job name
#SBATCH -e inference.err                        # error file
#SBATCH --gres=gpu:1                            # number of gpus 1, remove if you don't use gpu's
#SBATCH --mem=120gb                             # Job memory request
#SBATCH --time=1:00:00                          # time limit 1h
#SBATCH -o inferenceEns.out                     # output file
 
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

# run the python script -v is the model version -e the number of epochs -dd the data directory -rd the results directory -wd the directory with the pretrained weights
torchrun \
    --standalone \
    --nproc_per_node=1 /home/x3007104/thesis/scripts/inference.py \
    --version ensemble \
    --data-dir /deepstore/datasets/bms/hcc_study \
    --results-dir /home/x3007104/thesis/results \
    --weights-dir /home/x3007104/thesis/pretrained_models

