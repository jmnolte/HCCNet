#!/bin/bash
# parameters for slurm
#SBATCH -c 64                                   # number of cores, 64
#SBATCH --job-name=resnet18
#SBATCH -e resnet18.err                         # error file
#SBATCH --gres=gpu:4                            # number of gpus 4, remove if you don't use gpu's
#SBATCH --mem=120gb                             # Job memory request
#SBATCH --mail-user=j.m.nolte@students.uu.nl    # email address 
#SBATCH --mail-type=ALL                         # email when job starts, ends, and fails
#SBATCH -o resnet18.out                         # output file
 
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
torchrun --standalone --nproc_per_node=4 /home/x3007104/thesis/scripts/training.py -v resnet18 -e 10 -b 8 -dd /deepstore/datasets/bms/hcc_study -rd /home/x3007104/thesis/results -wd /home/x3007104/thesis/pretrained_models

