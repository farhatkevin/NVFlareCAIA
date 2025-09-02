#!/bin/bash

python3 synthea_llm_hf_fl_job.py \
 --client_ids train_1 train_2 \
 --data_path ${PWD}/synthea_data \
 --workspace_dir ${PWD}/workspace/hf_sft_multi \
 --job_dir ${PWD}/workspace/jobs/hf_sft_multi \
 --train_mode PEFT \
 --threads 1 \
 --gpu "[0],[1]" \
 --quantize_mode float16 \
 --num_rounds 1
