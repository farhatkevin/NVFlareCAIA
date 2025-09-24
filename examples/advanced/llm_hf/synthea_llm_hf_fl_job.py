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
import os

from src.hf_file_model_persistor import HFFileModelPersistor

from nvflare import FedJob, FilterType
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.app_opt.pt.quantization.dequantizer import ModelDequantizer
from nvflare.app_opt.pt.quantization.quantizer import ModelQuantizer
from nvflare.job_config.script_runner import BaseScriptRunner, ScriptRunner
from nvflare.private.fed.utils.fed_utils import split_gpus


def main():
    args = define_parser()
    train_script = "src/synthea_hf_sft_peft_fl.py"
    client_ids = args.client_ids
    num_clients = len(client_ids)
    # get the GPU assignments and ports
    gpus = split_gpus(args.gpu)
    gpus = [g.split(",") for g in gpus]
    ports = args.ports

    print(f"Clients: {client_ids}, GPUs: {gpus}, ports: {ports}")
    # make sure the number of GPUs matches the number of clients
    if len(gpus) != num_clients:
        raise ValueError(f"Number of GPUs ({len(gpus)}) does not match number of clients ({num_clients}).")
    # make sure the number of ports equal or greater than the number of clients
    if len(ports) < num_clients:
        raise ValueError(f"Number of ports ({len(ports)}) is less than number of clients ({num_clients}).")

    if args.threads:
        num_threads = args.threads
    else:
        num_threads = num_clients

    num_rounds = args.num_rounds
    workspace_dir = args.workspace_dir
    job_dir = args.job_dir
    model_name_or_path = args.model_name_or_path
    train_mode = args.train_mode
    message_mode = args.message_mode
    loss_log_file = args.loss_log_file
    eval_dump_file = args.eval_dump_file
    seed = args.seed

    # Create the FedJob
    if train_mode.lower() == "sft":
        job = FedJob(name="llm_hf_sft", min_clients=num_clients)
        output_path = "sft"
    elif train_mode.lower() == "peft":
        job = FedJob(name="llm_hf_peft", min_clients=num_clients)
        output_path = "peft"
    else:
        raise ValueError(f"Invalid train_mode: {train_mode}, only SFT and PEFT are supported.")

    # Define the FedAvg controller workflow and send to server
    controller = FedAvg(
        num_clients=num_clients,
        num_rounds=num_rounds,
    )
    job.to(controller, "server")

    if args.quantize_mode:
        # If using quantization, add quantize filters.
        quantizer = ModelQuantizer(quantization_type=args.quantize_mode)
        dequantizer = ModelDequantizer()
        job.to(quantizer, "server", tasks=["train"], filter_type=FilterType.TASK_DATA)
        job.to(dequantizer, "server", tasks=["train"], filter_type=FilterType.TASK_RESULT)

    # Define the model persistor and send to server
    if train_mode.lower() == "sft":
        # First send the model to the server
        job.to("src/hf_sft_model.py", "server")
        # Then send the model persistor to the server
        model_args = {"path": "src.hf_sft_model.CausalLMModel", "args": {"model_name_or_path": model_name_or_path}}
    elif train_mode.lower() == "peft":
        # First send the model to the server
        job.to("src/hf_peft_model.py", "server")
        # Then send the model persistor to the server
        model_args = {"path": "src.hf_peft_model.CausalLMPEFTModel", "args": {"model_name_or_path": model_name_or_path}}

    # change to using HFFileModelPersistor
    # job.to(PTFileModelPersistor(model=model_args, allow_numpy_conversion=False), "server", id="persistor")

    job.to(
        HFFileModelPersistor(
            model=model_args,
            allow_numpy_conversion=False,
            global_model_file_name="global_model",  # This is the directory name
            best_global_model_file_name="best_global_model",
            model_name_or_path=args.model_name_or_path,  # Pass the model name or path
        ),
        "server",
        id="persistor",
    )

    # Add model selection widget and send to server
    job.to(IntimeModelSelector(key_metric="eval_accuracy", negate_key_metric=False), "server", id="model_selector")

    # Send ScriptRunner to all clients
    for i in range(num_clients):
        client_id = client_ids[i]
        site_name = f"site-{client_id}"
        data_path_train = os.path.join(args.data_path, client_id, "training.jsonl")
        data_path_valid = os.path.join(args.data_path, client_id, "validation.jsonl")

        script_args = (
            f"--model_name_or_path {model_name_or_path} "
            f"--data_path_train {data_path_train} "
            f"--data_path_valid {data_path_valid} "
            f"--output_path {output_path} "
            f"--train_mode {train_mode} "
            f"--message_mode {message_mode} "
            f"--num_rounds {num_rounds} "
            f"--site_name {site_name}"
        )
        if loss_log_file:
            script_args += f" --loss_log_file {loss_log_file}"
        if eval_dump_file:
            script_args += f" --eval_dump_file {eval_dump_file}"
        if seed:
            script_args += f" --seed {seed}"

        if message_mode == "tensor":
            server_expected_format = "pytorch"
        elif message_mode == "numpy":
            server_expected_format = "numpy"
        else:
            raise ValueError(f"Invalid message_mode: {message_mode}, only numpy and tensor are supported.")



        # To customize timeouts, use BaseScriptRunner with a pre-configured executor. ScriptRunner does not accept an exector, but BaseScriptRunner does.
        # Note: BaseScriptRunner uses fixed component ids "pipe" and "launcher", which we match here.
        # This changes the timeouts properly, which we may need to modify depending on batch sizes, model size, etc.

        from nvflare.app_opt.pt.client_api_launcher_executor import PTClientAPILauncherExecutor

        executor = PTClientAPILauncherExecutor(
            pipe_id="pipe",
            launcher_id="launcher",
            peer_read_timeout=500.0,
            task_wait_timeout=500.0,
            launch_timeout=500.0,
            heartbeat_timeout=500.0,
            last_result_transfer_timeout=1000.0,
            server_expected_format=server_expected_format,
        )

        if len(gpus[i]) == 1:
            runner = BaseScriptRunner(
                script=train_script,
                script_args=script_args,
                server_expected_format=server_expected_format,
                launch_external_process=True,
                executor=executor,
            )
        else:
            runner = BaseScriptRunner(
                script=train_script,
                script_args=script_args,
                server_expected_format=server_expected_format,
                launch_external_process=True,
                command=f"python3 -m torch.distributed.run --nnodes=1 --nproc_per_node={len(gpus[i])} --master_port={ports[i]}",
                executor=executor,
            )
        job.to(runner, site_name, tasks=["train"])

        if args.quantize_mode:
            job.to(quantizer, site_name, tasks=["train"], filter_type=FilterType.TASK_RESULT)
            job.to(dequantizer, site_name, tasks=["train"], filter_type=FilterType.TASK_DATA)

        # Add additional parameters to clients
        #TODO: can add this if using nvflare 2.7rc, but with 2.6 will fail with AttributeError: 'dict' object has no attribute '__module__'. Did you mean: '__reduce__'?
        # client_params = {"submit_task_result_timeout": 300}
        # job.to(client_params, site_name)

    # Export the job
    print("job_dir=", job_dir)
    job.export_job(job_dir)

    # Run the job
    print("workspace_dir=", workspace_dir)
    print("num_threads=", num_threads)
    job.simulator_run(workspace_dir, threads=num_threads, gpu=args.gpu)


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--client_ids",
        nargs="+",
        type=str,
        default="",
        help="Clinet IDs, used to get the data path for each client",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=3,
        help="Number of rounds, default to 3",
    )
    parser.add_argument(
        "--workspace_dir",
        type=str,
        default="/tmp/nvflare/jobs/llm_hf/workdir",
        help="work directory, default to '/tmp/nvflare/jobs/llm_hf/workdir'",
    )
    parser.add_argument(
        "--job_dir",
        type=str,
        default="/tmp/nvflare/jobs/llm_hf/jobdir",
        help="directory for job export, default to '/tmp/nvflare/jobs/llm_hf/jobdir'",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/llama-3.1-8B-Instruct",
        help="model name or path",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="",
        help="root directory for training and validation data",
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="SFT",
        help="training mode, SFT or PEFT, default to SFT",
    )
    parser.add_argument(
        "--quantize_mode",
        type=str,
        default=None,
        help="quantization mode, default to None (no quantization)",
    )
    parser.add_argument(
        "--message_mode",
        type=str,
        default="numpy",
        help="message mode, numpy or tensor, default to numpy",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="number of threads to use for FL simulation, default to the number of clients",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="gpu assignments for simulating clients, comma separated, default to single gpu",
    )
    parser.add_argument(
        "--ports",
        nargs="+",
        default="7777",
        help="ports for the clients, default to one client 7777",
    )
    parser.add_argument(
        "--loss_log_file",
        type=str,
        help="Optional loss log file name to be written by each client; if relative, placed under output_path",
    )
    parser.add_argument(
        "--eval_dump_file",
        type=str,
        help="Optional JSONL file to append full eval generations each eval (relative path placed under output_path)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
