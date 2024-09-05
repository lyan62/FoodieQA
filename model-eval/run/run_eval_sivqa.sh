#!/bin/bash
#SBATCH --partition gpu
#SBATCH --nodes 1
#SBATCH --gres=gpu:3
#SBATCH --time 8:00:00
#SBATCH --job-name qwen
#SBATCH --output /ceph/hpc/data/d2024d05-018-users/wenyan/data/foodie/logs/%x.%j.out


cd $SLURM_SUBMIT_DIR # The path where this script was submitted from

echo $CUDA_VISIBLE_DEVICES # Prints the number of visible GPUs
echo $(date +"%Y-%m-%d %T") # Prints time and date

# source /ceph/hpc/home/euwenyanl/miniconda3/etc/profile.d/conda.sh
# Specify version of CUDA and GCC to use with conda env 
echo $LD_LIBRARY_PATH

echo $SLURMD_NODENAME $CUDA_VISIBLE_DEVICES
nvidia-smi

conda activate /ceph/hpc/home/euwenyanl/miniconda3/envs/foodie
python -c "import torch; print(torch.__version__)"


export HF_DATASETS_CACHE=/ceph/hpc/data/d2024d05-018-users/wenyan/cache
export DATA_DIR=/ceph/hpc/data/d2024d05-018-users/wenyan/data

cd scripts
# python eval_idefics_sivqa.py --template $1 \
#     --model_name "HuggingFaceM4/idefics2-8b" \
#     --cache_dir /ceph/hpc/data/d2024d05-018-users/wenyan/cache \
#     --data_dir $DATA_DIR/foodie/ \
#     --out_dir $DATA_DIR/foodie/results/sivqa_res \
#     --lang zh \
#     --eval_file sivqa_filtered_bi.json

python eval_qwen_sivqa.py --template $1 \
    --cache_dir /ceph/hpc/data/d2024d05-018-users/wenyan/cache \
    --data_dir $DATA_DIR/foodie/ \
    --out_dir $DATA_DIR/foodie/results/sivqa_res \
    --lang zh \
    --eval_file sivqa_filtered_bi.json
