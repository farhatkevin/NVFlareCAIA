#!/bin/bash

python scripts/evaluation.py \
 --model_name_or_path allenai/OLMo-2-0425-1B-Instruct \
 --data_path_valid /data/input/kf/NVFlare/examples/advanced/llm_hf/new_synthea_data/split_1/testing.jsonl /data/input/kf/NVFlare/examples/advanced/llm_hf/new_synthea_data/split_2/testing.jsonl