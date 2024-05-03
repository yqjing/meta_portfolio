#!/bin/bash
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=gpu_a100
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 15
#SBATCH --mem-per-cpu 4000

#SBATCH --job-name="run_5"
#SBATCH --mail-user=y5jing@uwaterloo.ca
#SBATCH --mail-type=end,fail   
#SBATCH --output=./outputs/%x-%j.out

module load anaconda3
cd ~/project2_test_2/examples/benchmarks_dynamic/incremental
python3 main.py











