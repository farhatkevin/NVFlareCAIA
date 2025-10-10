#!/bin/bash

# Example script to run exact training match evaluation
# Modify the paths and parameters below for your specific setup

set -e  # Exit on any error

# Configuration
# MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
# MODEL_NAME="allenai/OLMo-2-0425-1B-Instruct"
# MODEL_NAME="/data/input/kf/NVFlare/examples/advanced/llm_hf/merged_model/round1_final/merged_sma"
# MODEL_NAME="/data/input/kf/NVFlare/examples/advanced/llm_hf/workspace/hf_sft_multi_2/server/simulate_job/app_server/round_1_model_hf"
# MODEL_NAME="aaditya/Llama3-OpenBioLLM-70B"
# MODEL_NAME="/data/input/kf/NVFlare/examples/advanced/llm_hf/workspace/synthea_combined/server/simulate_job/app_server/round_1_model_hf"
MODEL_NAME="/data/input/kf/NVFlare/examples/advanced/llm_hf/workspace/hf_sft_multi_test_constrained_eval_test_set/site-split_1/simulate_job/app_site-split_1/sft/checkpoint-664"
# MODEL_NAME="/data/input/kf/NVFlare/examples/advanced/llm_hf/workspace/hf_sft_multi_test_constrained_eval_test_set/server/simulate_job/app_server/round_1_model_hf"
DATASET_PATHS="/data/input/kf/NVFlare/examples/advanced/llm_hf/new_synthea_data/split_1/testing.jsonl /data/input/kf/NVFlare/examples/advanced/llm_hf/new_synthea_data/split_2/testing.jsonl"
# DATASET_PATHS="/data/input/kf/NVFlare/examples/advanced/llm_hf/new_synthea_data/split_1/validation.jsonl /data/input/kf/NVFlare/examples/advanced/llm_hf/new_synthea_data/split_2/validation.jsonl"
# DATASET_PATHS="/data/input/kf/NVFlare/examples/advanced/llm_hf/new_synthea_data/split_1_training_eval_subset.jsonl"
BATCH_SIZE=16
NUM_EXAMPLES=-1  # Set to 200 to match training, or -1 for all examples
SEED=42
# Create model name for directory (replace / and special chars with _)
MODEL_DIR_NAME=$(echo "$MODEL_NAME" | sed 's/[\/:]/_/g')
OUTPUT_DIR="local_model_results_${MODEL_DIR_NAME}_$(date +%Y%m%d_%H%M%S)"

echo "=============================================="
echo "Running Exact Training Match MCQ Evaluation"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "Dataset Files: $DATASET_PATHS"
echo "Batch Size: $BATCH_SIZE"
echo "Number of Examples: $NUM_EXAMPLES"
echo "Seed: $SEED"
echo "Output Directory: $OUTPUT_DIR"
echo "=============================================="

# Create output directory
mkdir -p "evaluation_output/$OUTPUT_DIR"

# Run the evaluation with exact training match methodology
python scripts/eval/evaluate_local_model_constrained.py \
    --model "$MODEL_NAME" \
    --dataset $DATASET_PATHS \
    --batch-size "$BATCH_SIZE" \
    --output-dir "$OUTPUT_DIR" \
    --num-examples "$NUM_EXAMPLES" \
    --seed "$SEED" \
    --choices A B C D E

# Check if evaluation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Evaluation completed successfully!"
    echo "=============================================="
    echo "Results saved in: evaluation_output/$OUTPUT_DIR"
    echo ""
    echo "Files created:"
    ls -la "evaluation_output/$OUTPUT_DIR"
    echo ""
    echo "Quick metrics summary:"
    if [ -f "evaluation_output/$OUTPUT_DIR/metrics.csv" ]; then
        echo "Accuracy: $(grep accuracy "evaluation_output/$OUTPUT_DIR/metrics.csv" | cut -d',' -f2)"
    fi
else
    echo ""
    echo "=============================================="
    echo "Evaluation failed!"
    echo "=============================================="
    exit 1
fi