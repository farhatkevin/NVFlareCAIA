# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import os

# Add deterministic seed for reproducibility illustration
import random
import shutil
from functools import partial

import datasets
import numpy as np
import torch
import torch.distributed as dist
from accelerate import PartialState
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, utils
from training_utils import (
    compute_metrics,
    preprocess_logits_for_metrics,
    compute_loss_mcq5,
    create_loss_logger,
    state_dict_fingerprint,
    model_fingerprint,
    filter_by_length_messages,
    format_instruction,
    wrap_compute_metrics,
    WandbMetricsLogger,
    EvalDumpRecordedCallback,
)

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, trainer_utils
import transformers 
print(f"Transformers version: {transformers.__version__}")


from trl import SFTConfig, SFTTrainer

import nvflare.client as flare

MAX_SEQ_LENGTH = 4096

# Add callback to stop at each epoch
class StopCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        control.should_training_stop = True

# set deterministic seed for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def setup_distributed_training():
    """Setup distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Set device
        torch.cuda.set_device(local_rank)

        print(f"Distributed training initialized: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        return rank, world_size, local_rank
    else:
        print("No distributed training environment detected, running in single GPU mode")
        return 0, 1, 0


def cleanup_distributed_training():
    """Cleanup distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--data_path_train",
        type=str,
        default="./synthea_data/train.jsonl",
    )
    parser.add_argument(
        "--data_path_valid",
        type=str,
        default="./synthea_data/test.jsonl",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./workspace_federated/llama-3.1-8B-Instruct-synthea-sft",
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="SFT",
        help="training mode, SFT or PEFT, default to SFT",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="learning rate scheduler type, default to 'constant'",
    )
    parser.add_argument(
        "--message_mode",
        type=str,
        default="numpy",
        help="message mode, numpy or tensor, default to numpy",
    )
    parser.add_argument("--local_epoch", type=int, default=1)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument(
        "--site_name",
        type=str,
        default=os.getenv("NVFLARE_SITE_NAME", "unknown"),
        help="NVFlare site name for per-site loss logging",
    )
    parser.add_argument(
        "--loss_log_file",
        type=str,
        default=None,
        help="Optional explicit file to write loss logs; defaults to <output_path>/loss_<site>.log",
    )
    parser.add_argument(
        "--eval_dump_file",
        type=str,
        default=None,
        help="Optional JSONL file to append full eval generations each eval",
    )
    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed_training()

    # Set up device for DDP
    device_string = PartialState().process_index
    device_map = {"": device_string}

    # If output path exists, remove it (only on main process)
    if local_rank == 0:
        try:
            print(f"Attempting to remove output path {args.output_path}.")
            shutil.rmtree(args.output_path)
        except FileNotFoundError:
            print(f"Output path {args.output_path} does not exist, skipping removal.")

    # Wait for main process to finish cleanup
    if dist.is_initialized():
        dist.barrier()

    # TODO: remove select after testing
    dataset_train = (
        datasets.load_dataset("json", data_files=args.data_path_train, split="train").shuffle().select(range(10000))
        # datasets.load_dataset("json", data_files=args.data_path_train, split="train").shuffle()

    )
    dataset_valid = (
        datasets.load_dataset("json", data_files=args.data_path_valid, split="train").shuffle().select(range(50))
        # datasets.load_dataset("json", data_files=args.data_path_valid, split="train").shuffle()

    )

    # Model configs
    model_name_or_path = args.model_name_or_path
    peft_config = None

    # Load tokenizer (moved up to be available for filtering)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if tokenizer has a chat template
    has_chat_template = hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None

    if local_rank == 0:
        if has_chat_template:
            print(f"Using existing chat template from {model_name_or_path}")
            print(f"Chat template preview: {tokenizer.chat_template[:200]}...")
        else:
            print(f"No chat template found for {model_name_or_path}, using fallback template")

    # Only set fallback if no template exists
    if not has_chat_template:
        # Fallback chat template for models without one
        tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'user' %}{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'assistant' %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

    dataset_train = dataset_train.map(format_instruction, batched=True, remove_columns=dataset_train.column_names)
    dataset_valid = dataset_valid.map(format_instruction, batched=True, remove_columns=dataset_valid.column_names)

    # Then apply filtering with debug info
    original_train_size = len(dataset_train)
    original_valid_size = len(dataset_valid)

    # Debug: Check a sample before filtering
    if local_rank == 0 and len(dataset_train) > 0:
        sample = dataset_train[0]
        print(f"Sample before filtering: {sample.keys()}")
        if "messages" in sample:
            print(
                f"Sample messages structure: {sample['messages'][:200] if isinstance(sample['messages'], str) else sample['messages']}"
            )
            try:
                formatted = tokenizer.apply_chat_template(
                    sample["messages"], tokenize=False, add_generation_prompt=False
                )
                tokens = tokenizer.encode(formatted)
                print(f"Sample token length: {len(tokens)} (max: {MAX_SEQ_LENGTH})")
            except Exception as e:
                print(f"Error processing sample: {e}")

    dataset_train = dataset_train.filter(lambda x: filter_by_length_messages(x, tokenizer, MAX_SEQ_LENGTH))
    dataset_valid = dataset_valid.filter(lambda x: filter_by_length_messages(x, tokenizer, MAX_SEQ_LENGTH))

    # Print dataset info
    if local_rank == 0:
        print(f"Filtered dataset sizes:")
        print(f"  Training: {original_train_size} -> {len(dataset_train)}")
        print(f"  Validation: {original_valid_size} -> {len(dataset_valid)}")
        if len(dataset_train) == 0:
            print("WARNING: Training dataset is empty after filtering!")

    # record every 5% of the dataset
    batch_size = 4
    gra_accu_steps = 10
    logging_steps = int(len(dataset_train) / (20 * batch_size * gra_accu_steps))
    if local_rank == 0:
        print(f"logging_steps: {logging_steps}")

    # (No response_template used in original config)

    # Load model with device_map
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        use_cache=False,
        torch_dtype=torch.bfloat16,
    )
    torch.set_default_dtype(default_dtype)

    # Train mode
    if args.train_mode.lower() == "sft":
        train_mode = 0
    elif args.train_mode.lower() == "peft":
        train_mode = 1
    else:
        raise ValueError(f"Invalid train_mode: {args.train_mode}, only SFT and PEFT are supported.")

    # PEFT specific
    if train_mode:
        # PEFT configs
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
    model.config.pretraining_tp = 1

    # Training arguments
    train_args = SFTConfig(
        output_dir=args.output_path,
        # Using callback, stop at each epoch, so specify num_train_epochs
        # the same as the total epoch in one-call training
        num_train_epochs=args.local_epoch * args.num_rounds,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=gra_accu_steps,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # optimizers using bitsandbytes like "paged_adamw_32bit" have an issue with
        # multi-gpu training, to be consistent, use regular optimizer
        optim="adamw_torch",
        logging_steps=20,
        save_strategy="epoch",
        learning_rate=5e-6,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        # use cosine_with_restarts scheduler to check the iterative behavior
        lr_scheduler_type=args.lr_scheduler,
        lr_scheduler_kwargs={"num_cycles": 2},
        disable_tqdm=True,
        save_total_limit=2,
        # safetensors will remove shared layers, e.g. lm_head.weight
        # disable for local checkpointing
        eval_strategy="steps",
        # eval_on_start=True,
        eval_steps=50,
        save_safetensors=False,
        seed=0,
        data_seed=0,
        # Multi-GPU specific settings
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,
        # Prompt Completion w/ Mitchell
        completion_only_loss=True,
        metric_for_best_model="eval_accuracy",
        max_length=MAX_SEQ_LENGTH,
        report_to=["wandb"],
    )

    ##########################################################
    ###   extra code for logging that we can remove later  ###
    ##########################################################

    # Set up per-site loss logger (rank 0 only writes)
    site_name = args.site_name
    loss_logger = create_loss_logger(
        site_name=site_name, output_path=args.output_path, loss_log_file=args.loss_log_file, local_rank=local_rank
    )
    wandb_cb = WandbMetricsLogger(site_name=site_name, local_rank=local_rank)

    # Prepare compute_metrics wrapper and optional eval dump recorder
    base_compute = partial(compute_metrics, tokenizer=tokenizer, verbose=True)
    compute_fn = base_compute
    eval_dump_cb = None
    if args.eval_dump_file:
        compute_fn = wrap_compute_metrics(base_compute)
        dump_path = (
            args.eval_dump_file
            if os.path.isabs(args.eval_dump_file)
            else os.path.join(args.output_path, args.eval_dump_file)
        )
        eval_dump_cb = EvalDumpRecordedCallback(out_path=dump_path, local_rank=local_rank, tokenizer=tokenizer)

    ##########################################################

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_valid,
        peft_config=peft_config,
        processing_class=tokenizer,
        compute_metrics=compute_fn,
        # compute_loss_func=compute_loss_mcq5,
        compute_loss_func=lambda outputs, labels, num_items_in_batch=None: compute_loss_mcq5(outputs, labels, num_items_in_batch, tokenizer=tokenizer),
        # need to pass in tokenizer to get encoded A–E labels, so using lambda function
        preprocess_logits_for_metrics=lambda logits, labels: preprocess_logits_for_metrics(logits, labels, tokenizer),
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        args=train_args,
        callbacks=[StopCallback(), loss_logger, wandb_cb]
        + ([eval_dump_cb] if eval_dump_cb is not None else []),
    )

    # Untouched from original

    # initializes NVFlare client API
    flare.init()

    # Train federated rounds
    # start with global model at the beginning of each round
    while flare.is_running():
        # receives golobal model from NVFlare (only on main process)
        if local_rank == 0:
            input_model = flare.receive()
            curr_round = input_model.current_round
            print(f"current_round={curr_round}")
            # Update the key name received from global model if using model def file
            global_model = copy.deepcopy(input_model.params)
            for key in list(global_model.keys()):
                global_model[key.replace("model.", "", 1)] = global_model.pop(key)
        else:
            curr_round = None
            global_model = None

        # broadcast current round and global_model to all processes
        if dist.is_initialized():
            curr_round_list = [curr_round]
            global_model_list = [global_model]
            dist.broadcast_object_list(curr_round_list, src=0)
            dist.broadcast_object_list(global_model_list, src=0)
            curr_round = curr_round_list[0]
            global_model = global_model_list[0]

        if dist.is_initialized():
            dist.barrier()

        # Load state dict
        if local_rank == 0 and global_model is not None:
            # Fingerprint the received global model before loading
            fp_recv = state_dict_fingerprint(global_model, name_hint=f"global_r{curr_round}")
            print(f"[Round {curr_round}] Received global model fp: {fp_recv}")
            # Also show PEFT-only view of received global to compare apples-to-apples with loaded PEFT state
            try:
                peft_only_sd = {k: v for k, v in global_model.items() if "lora_" in k}
                if len(peft_only_sd) > 0:
                    fp_recv_peft = state_dict_fingerprint(peft_only_sd, name_hint=f"global_r{curr_round}_peft_only")
                    print(f"[Round {curr_round}] Received global (PEFT-only) fp: {fp_recv_peft}")
            except Exception as e:
                print(f"[Round {curr_round}] PEFT-only fingerprint of received global failed: {e}")
        if train_mode:
            set_peft_model_state_dict(trainer.model, global_model)
        else:
            trainer.model.load_state_dict(global_model)
        # Wait for main process to finish model loading
        if dist.is_initialized():
            dist.barrier()
        if local_rank == 0:
            # Fingerprint the model in memory after loading global weights
            fp_loaded = model_fingerprint(trainer.model, peft_only=bool(train_mode))
            print(f"[Round {curr_round}] Model after load fp: {fp_loaded}")

        # Update current round for logging, then evaluate the global model
        try:
            loss_logger.set_round(curr_round)
            wandb_cb.set_round(curr_round)
        except Exception:
            pass
        # Evaluate the global model (reset debug so first eval batch prints A–E logits)
        try:
            preprocess_logits_for_metrics.debug_count = 0
        except Exception:
            pass
        eval_results = trainer.evaluate()
        eval_loss = float(eval_results["eval_loss"])
        eval_accuracy = eval_results.get("eval_accuracy", 0.0)
        if local_rank == 0:
            print(f"Evaluation - Loss: {eval_loss:.4f}, Accuracy: {eval_accuracy:.4f}")

        # Train
        if curr_round == 0:
            # First round, start from pretrained model
            for epoch in range(args.local_epoch):
                print(f"Training local epoch {epoch + 1}/{args.local_epoch}")
                # train for one epoch
                if epoch == 0:
                    trainer.train()
                else:
                    # continue training
                    trainer.train(resume_from_checkpoint=True)
        else:
            # replace local resume weights with global weights (only on main process)
            if local_rank == 0:
                resume_from_checkpoint_folder = trainer_utils.get_last_checkpoint(trainer.args.output_dir)
                if train_mode:
                    # PEFT model small, directly save via torch.save
                    # TODO: remove this
                    print("resume_from_checkpoint_folder:", resume_from_checkpoint_folder)
                    print("utils.WEIGHTS_NAME:", utils.WEIGHTS_NAME)
                    resume_model_file_path = os.path.join(resume_from_checkpoint_folder, utils.WEIGHTS_NAME)
                    torch.save(global_model, resume_model_file_path)
                else:
                    # SFT model can be large, save via HF API
                    # Disable safetensor for now
                    trainer.model.save_pretrained(
                        resume_from_checkpoint_folder, state_dict=global_model, safe_serialization=False
                    )

            # Wait for main process to finish saving before continuing
            if dist.is_initialized():
                dist.barrier()

            # continue training
            # as we used callback, no need to increment num_train_epochs
            for epoch in range(args.local_epoch):
                print(f"Training local epoch {epoch + 1}/{args.local_epoch}")
                trainer.train(resume_from_checkpoint=True)

        # Wait for all process to finish training before continuing
        if dist.is_initialized():
            dist.barrier()
        if local_rank == 0:
            # Fingerprint the locally trained model
            fp_trained = model_fingerprint(trainer.model, peft_only=bool(train_mode))
            print(f"[Round {curr_round}] Model after local training fp: {fp_trained}")

        # compose output model to send back to server (only on main process)
        if local_rank == 0:
            if train_mode:
                # PEFT, load PEFT part from trainer model
                out_param = get_peft_model_state_dict(trainer.model)
            else:
                # SFT, load whole model state_dict
                out_param = trainer.model.state_dict()

            # update the key name sent to global model
            if not train_mode:
                for key in list(out_param.keys()):
                    out_param["model." + key] = out_param.pop(key).cpu()

            if args.message_mode.lower() == "numpy":
                # cast out_param to float32 preparing for communication with numpy
                # otherwise do nothing
                out_param = {k: v.to(torch.float32) for k, v in out_param.items()}

            # print the dict size
            print(f"In total {len(out_param.keys())} params to be sent to server.")
            # Fingerprint the outgoing params (what we send back)
            fp_out = state_dict_fingerprint(out_param, name_hint=f"client_out_r{curr_round}")
            print(f"[Round {curr_round}] Outgoing params fp: {fp_out}")

            # construct trained FL model
            output_model = flare.FLModel(
                params=out_param,
                metrics={"eval_loss": eval_loss, "eval_accuracy": eval_accuracy},
                meta={"NUM_STEPS_CURRENT_ROUND": trainer.train_dataset.num_rows},
            )
            # send model back to NVFlare
            flare.send(output_model)


if __name__ == "__main__":
    main()
