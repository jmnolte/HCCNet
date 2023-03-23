#!/bin/bash
# parameters for slurm
#SBATCH -c 8                                    # number of cores, 1
#SBATCH -e resnet10.err                         # error file
#SBATCH --gres=gpu:1                            # number of gpus 4, remove if you don't use gpu's
#SBATCH --mem=120gb                             # Job memory request
#SBATCH --mail-user=j.m.nolte@students.uu.nl    # email address 
#SBATCH --mail-type=ALL                         # email when job starts, ends, and fails
#SBATCH --time=1:00:00                          # time limit 1h
#SBATCH -o resnet10.out                         # output file
 
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
python /home/x3007104/thesis/scripts/training.py -v resnet10 -e 10 -b 32 -dd /deepstore/datasets/bms/hcc_study -rd /home/x3007104/thesis/results -wd /home/x3007104/thesis/pretrained_models

