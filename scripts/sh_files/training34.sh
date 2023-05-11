#!/bin/bash
# parameters for slurm
#SBATCH -c 32                                   # number of cores, 32
#SBATCH --job-name=resnet34                     # job name
#SBATCH -e resnet34.err                         # error file
#SBATCH --gres=gpu:4                            # number of gpus 4
#SBATCH --mem=120gb                             # Job memory request
#SBATCH -o resnet34.out                         # output file
 
# show actual node in output file
hostname
 
# load all required software modules. This may be different according to the HPC configuration used.
module load nvidia/cuda-11.6
module load miniconda3/3.8
# activate base conda environment
source activate base
# activate your individual conda environment (in this case named hcc-gpu)
conda activate hcc-gpu
 
# show the number of available GPUs
echo "Gpu devices                 : "$CUDA_VISIBLE_DEVICES
echo "Starting worker: "

# training script for resnet34 using fault tolerant training. Script returns updated set of model weights 
# and diagnostic plots of model loss and F1-score plotted against the number of training epochs. 

# Ro run the script, note that all paths need to be adjusted according to the local file system. 
# Note also that if access to a high performance cluster is not available, please execute the 
# below command in your command line. If executed from the command line the --standalone flag and the 
# number of processes per node flag should be removed. 

torchrun \
    --standalone \
    --nproc_per_node=4 /path_to_script_directory/training.py \
    --version resnet34 \
    --pretrained \
    --weighted-sampler \
    --epochs 10 \
    --batch-size 4 \
    --accum-steps 8 \
    --learning-rate 1e-2 \
    --weight-decay 1e-5 \
    --data-dir /path_to_data_directory \
    --results-dir /path_to_output_directory \
    --weights-dir /path_to_pretrained_weights