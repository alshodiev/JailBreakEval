#!/bin/bash

#SBATCH -J extr
#SBATCH -c 1                             # Request 1 core
#SBATCH -t 0-05:00                       # Runtime in D-HH:MM format
#SBATCH -p g52xlarge-on
#SBATCH --gres=gpu:1
#SBATCH --mem=30G                        
#SBATCH -o job_err_out/extr.out
#SBATCH -e job_err_out/extr.err
#SBATCH --array=0 #Array jobs for hyperparameter combinations


python embedding_extr.py \    
                   --batch_size 16 \
                   --model_name roberta-large-openai-detector
