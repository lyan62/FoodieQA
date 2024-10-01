#!/bin/bash
#SBATCH --partition gpu
#SBATCH --nodes 1
#SBATCH --gres=gpu:3
#SBATCH --time 8:00:00
#SBATCH --job-name qwen
#SBATCH --output data/foodie/logs/%x.%j.out


cd $SLURM_SUBMIT_DIR # The path where this script was submitted from

echo $CUDA_VISIBLE_DEVICES # Prints the number of visible GPUs
echo $(date +"%Y-%m-%d %T") # Prints time and date

# source /ceph/hpc/home/euwenyanl/miniconda3/etc/profile.d/conda.sh
# Specify version of CUDA and GCC to use with conda env 
echo $LD_LIBRARY_PATH

echo $SLURMD_NODENAME $CUDA_VISIBLE_DEVICES
nvidia-smi

python -c "import torch; print(torch.__version__)"


export HF_DATASETS_CACHE=cache_path
export DATA_DIR=data_path

cd scripts
python eval_idefics_sivqa.py --template $1 \
    --model_name "HuggingFaceM4/idefics2-8b" \
    --cache_dir $HF_DATASETS_CACHE \
    --data_dir $DATA_DIR/foodie/ \
    --out_dir $DATA_DIR/foodie/results/sivqa_res \
    --lang zh \
    --eval_file sivqa_tiqy.json

# python eval_mivqa_en.py --template $1 \
#     --model_name "microsoft/Phi-3-vision-128k-instruct" \
#     --cache_dir $HF_DATASETS_CACHE \
#     --data_dir $DATA_DIR/foodie/ \
#     --out_dir $DATA_DIR/foodie/results/mivqa_res \
#     --lang zh \
#     --eval_file mivqa_tiqy.json
