#!/bin/bash
#SBATCH --account dd-23-107
#SBATCH --partition qgpu
#SBATCH --nodes 1
#SBATCH --gpus 3
#SBATCH --time 1:00:00
#SBATCH --job-name idefics2
#SBATCH --output /scratch/project/dd-23-107/wenyan/data/foodie/logs/%x.%j.out
#SBATCH --exclude=acn07


cd $SLURM_SUBMIT_DIR # The path where this script was submitted from

echo $CUDA_VISIBLE_DEVICES # Prints the number of visible GPUs
echo $(date +"%Y-%m-%d %T") # Prints time and date

source /home/it4i-liyan/miniconda3/etc/profile.d/conda.sh


# Specify version of CUDA and GCC to use with conda env 
echo $LD_LIBRARY_PATH

echo $SLURMD_NODENAME $CUDA_VISIBLE_DEVICES
nvidia-smi

conda activate /scratch/project/dd-23-107/wenyan/envs/foodie
python -c "import torch; print(torch.__version__)"


export HF_DATASETS_CACHE=/scratch/project/dd-23-107/wenyan/cache
export PROMPT=$1

cd scripts
python eval_idefics2.py --prompt PROMPT