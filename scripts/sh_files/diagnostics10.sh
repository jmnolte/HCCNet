#!/bin/bash
# parameters for slurm
#SBATCH -c 32                                   # number of cores, 32
#SBATCH --job-name=diagnostics10                # job name
#SBATCH -e diagnostics.err                      # error file
#SBATCH --gres=gpu:1                            # number of gpus, 4
#SBATCH --mem=120gb                             # Job memory request
#SBATCH -o diagnostics10.out                    # output file
 
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

# diagnostics script for resnet10 using fault tolerant processing. Script returns occlusion sensitivity
# mapping of the model's activations for two example images (i.e., one with HCC and one without HCC) as 
# depicted in figure 2. Additionally, the script returns the model's receiver operating characteristic and 
# precision-recall curves as depicted in figure 3.

# To run the script, note that all paths need to be adjusted according to the local file system. 
# Note also that if access to a high performance cluster is not available, please only execute the 
# below command in your command line. If executed from the command line the --standalone flag and the 
# number of processes per node flag should be removed. 

torchrun \
    --standalone \
    --nproc_per_node=1 /path_to_script_directory/diagnostics.py \
    --version resnet10 \
    --occ-sens \
    --data-dir /path_to_data_directory \
    --results-dir /path_to_output_directory \
    --weights-dir /path_to_pretrained_weights