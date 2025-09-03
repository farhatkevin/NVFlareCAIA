import re
import numpy as np
import torch
from typing import Dict, Any, Callable
from transformers import TrainerCallback


def compute_metrics(eval_preds, tokenizer, verbose=False) -> Dict[str, float]:
    """
    Dummy compute_metrics function - returns placeholder accuracy.
    TODO: Replace with actual implementation later.
    """
    return {"accuracy": 0.5}


def extract_single_letter_answer(text: str) -> str:
    """
    Dummy extract_single_letter_answer function - returns 'A'.
    TODO: Replace with actual implementation later.
    """
    return 'A'


def filter_by_length(example, tokenizer, format_instruction: Callable, max_seq_length: int):
    """
    Filter dataset examples that exceed the maximum sequence length.
    
    Args:
        example: Dataset example with 'prompt' and 'completion' fields
        tokenizer: The tokenizer used for the model
        format_instruction: Function to format examples into instruction format
        max_seq_length: Maximum sequence length in tokens
        
    Returns:
        Boolean indicating whether to keep the example (True if under limit)
    """
    # Since format_instruction now returns a dict, manually concatenate for length check
    full_text = example["prompt"] + "\n\n" + example["completion"]
    tokens = tokenizer(full_text, truncation=False, return_tensors=None)
    return len(tokens["input_ids"]) < max_seq_length - 1


# ============================================================================
# DEBUGGING UTILITIES - For checking loss calculation and model I/O
# ============================================================================

class DebuggingCallback(TrainerCallback):
    """Callback to monitor training loss and learning rate during training"""
    # def on_log(self, args, state, control, model=None, logs=None, **kwargs):
    #     if logs:
    #         print(f"Step {state.global_step}: Loss = {logs.get('train_loss', 'N/A')}")
    #         print(f"Learning rate: {logs.get('learning_rate', 'N/A')}")
    
    # def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
    #     if logs:
    #         print(f"Eval step {state.global_step}: Eval loss = {logs.get('eval_loss', 'N/A')}")


def debug_data_collation(trainer, tokenizer, num_samples=2):
    """Debug what the model actually sees during training"""
    print("=" * 80)
    print("DEBUGGING DATA COLLATION AND MODEL PREDICTIONS")
    print("=" * 80)
    
    # Get a small batch from train dataloader
    dataloader = trainer.get_train_dataloader()
    batch = next(iter(dataloader))
    
    # Get model predictions for this batch
    model = trainer.model
    model.eval()
    
    with torch.no_grad():
        # Forward pass to get logits
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch.get('attention_mask'))
        logits = outputs.logits
        # Get predicted token IDs
        predicted_ids = torch.argmax(logits, dim=-1)
    
    for i in range(min(num_samples, len(batch['input_ids']))):
        input_ids = batch['input_ids'][i]
        labels = batch['labels'][i] if 'labels' in batch else None
        pred_ids = predicted_ids[i]
        
        print(f"\nSample {i+1}:")
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Labels shape: {labels.shape if labels is not None else 'None'}")
        
        # Decode the full sequence
        full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        print(f"Full sequence: {full_text[:500]}...")
        
        if labels is not None:
            # Show which tokens have loss calculated (not -100)
            loss_mask = labels != -100
            num_loss_tokens = loss_mask.sum().item()
            total_tokens = len(labels)
            print(f"Loss calculated on {num_loss_tokens}/{total_tokens} tokens ({num_loss_tokens/total_tokens:.1%})")
            
            if len(loss_mask) > 0 and num_loss_tokens > 0:
                # Ground truth completion tokens
                ground_truth_tokens = labels[loss_mask]
                ground_truth_text = tokenizer.decode(ground_truth_tokens, skip_special_tokens=False)
                print(f"Ground truth completion: '{ground_truth_text}'")
                
                # Model's predicted completion tokens (at the same positions)
                predicted_completion_tokens = pred_ids[loss_mask]
                predicted_text = tokenizer.decode(predicted_completion_tokens, skip_special_tokens=False)
                print(f"Model predicted completion: '{predicted_text}'")
                
                # Show where the completion starts
                first_loss_idx = torch.where(loss_mask)[0][0].item()
                completion_start_text = tokenizer.decode(input_ids[:first_loss_idx+5], skip_special_tokens=False)
                print(f"Completion starts around: '...{completion_start_text[-100:]}'")
                
            elif num_loss_tokens == 0:
                print("WARNING: No tokens with loss! This sample won't contribute to training.")
        
        print("-" * 40)


def test_generation(model, tokenizer, test_prompt, max_length=50):
    """Test model generation capability"""
    model.eval()
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=inputs.input_ids.shape[1] + max_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text