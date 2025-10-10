#!/bin/bash

# Example script to run constrained evaluation on API-based models
# Supports OpenAI and Anthropic models

set -e  # Exit on any error

# Configuration
DATASET_PATHS="/data/input/kf/NVFlare/examples/advanced/llm_hf/new_synthea_data/split_1/testing.jsonl /data/input/kf/NVFlare/examples/advanced/llm_hf/new_synthea_data/split_2/testing.jsonl"
BATCH_SIZE=5  # Smaller batch size for API rate limits

# Model configurations - uncomment the one you want to test
# OpenAI models
PROVIDER="openai"
MODEL_NAME="gpt-5-2025-08-07"
# MODEL_NAME="gpt-5-nano-2025-08-07"
# MODEL_NAME="gpt-4o-mini"
# MODEL_NAME="gpt-4o"
# MODEL_NAME="gpt-3.5-turbo"

# Anthropic models
# PROVIDER="anthropic"
# MODEL_NAME="claude-sonnet-4-20250514"
# MODEL_NAME="claude-3-5-sonnet-20241022"
# MODEL_NAME="claude-3-opus-20240229"

# API Key - set your API key as environment variable
if [ "$PROVIDER" = "openai" ]; then
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "Error: OPENAI_API_KEY environment variable not set"
        echo "Please set it with: export OPENAI_API_KEY='your-key-here'"
        exit 1
    fi
    API_KEY="$OPENAI_API_KEY"
elif [ "$PROVIDER" = "anthropic" ]; then
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo "Error: ANTHROPIC_API_KEY environment variable not set"
        echo "Please set it with: export ANTHROPIC_API_KEY='your-key-here'"
        exit 1
    fi
    API_KEY="$ANTHROPIC_API_KEY"
fi

# Create model name for directory (replace / and special chars with _)
MODEL_DIR_NAME=$(echo "${PROVIDER}_${MODEL_NAME}" | sed 's/[\/:]/_/g' | sed 's/-/_/g')
OUTPUT_DIR="api_model_results_${MODEL_DIR_NAME}_$(date +%Y%m%d_%H%M%S)"

echo "=============================================="
echo "Running API Constrained MCQ Evaluation"
echo "=============================================="
echo "Provider: $PROVIDER"
echo "Model: $MODEL_NAME"
echo "Dataset Files: $DATASET_PATHS"
echo "Batch Size: $BATCH_SIZE"
echo "Output Directory: evaluation_output/$OUTPUT_DIR"
echo "=============================================="

# Output directory will be created by the Python script

# Run the evaluation
python scripts/eval/evaluate_api_model_constrained.py \
    --provider "$PROVIDER" \
    --model "$MODEL_NAME" \
    --api-key "$API_KEY" \
    --dataset $DATASET_PATHS \
    --batch-size "$BATCH_SIZE" \
    --output-dir "$OUTPUT_DIR" \
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