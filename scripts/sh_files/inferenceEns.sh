#!/bin/bash
# parameters for slurm
#SBATCH -c 32                                   # number of cores, 32
#SBATCH --job-name=inferenceEns                 # job name
#SBATCH -e inference.err                        # error file
#SBATCH --gres=gpu:1                            # number of gpus, 1
#SBATCH --mem=120gb                             # Job memory request
#SBATCH -o inferenceEns.out                     # output file
 
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

# inference script for the ensemble using fault tolerant processing. Script returns model performance metrics
# on the study's test set as depicted in table 2. 

# To run the script, note that all paths need to be adjusted according to the local file system. 
# Note also that if access to a high performance cluster is not available, please only execute the 
# below command in your command line. If executed from the command line the --standalone flag and the 
# number of processes per node flag should be removed. 

torchrun \
    --standalone \
    --nproc_per_node=1 /path_to_script_directory/inference.py \
    --version ensemble \
    --data-dir /path_to_data_directory \
    --results-dir /path_to_output_directory \
    --weights-dir /path_to_pretrained_weights