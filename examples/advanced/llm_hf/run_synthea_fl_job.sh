#!/bin/bash

python3 synthea_llm_hf_fl_job.py \
 --client_ids split_1 split_2 \
 --data_path ${PWD}/new_synthea_data \
 --workspace_dir ${PWD}/workspace/hf_sft_multi \
 --job_dir ${PWD}/workspace/jobs/hf_sft_multi \
 --train_mode SFT \
 --threads 1 \
 --gpu "[0],[1]" \
 --quantize_mode float16 \
 --message_mode tensor \
 --num_rounds 2 \
 --model_name_or_path meta-llama/Llama-3.2-1B-Instruct\
 --loss_log_file loss.txt \
 --eval_dump_file eval_dump.jsonl

