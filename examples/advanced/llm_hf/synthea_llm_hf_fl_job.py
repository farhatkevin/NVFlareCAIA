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

from nvflare import FedJob, FilterType
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.app_opt.pt.quantization.dequantizer import ModelDequantizer
from nvflare.app_opt.pt.quantization.quantizer import ModelQuantizer
from nvflare.job_config.script_runner import ScriptRunner, BaseScriptRunner
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
    job.to(PTFileModelPersistor(model=model_args, allow_numpy_conversion=False), "server", id="persistor")

    # Add model selection widget and send to server
    job.to(IntimeModelSelector(key_metric="eval_loss", negate_key_metric=True), "server", id="model_selector")

    # Send ScriptRunner to all clients
    for i in range(num_clients):
        client_id = client_ids[i]
        site_name = f"site-{client_id}"
        data_path_train = os.path.join(args.data_path, client_id, "training.jsonl")
        data_path_valid = os.path.join(args.data_path, client_id, "validation.jsonl")

        script_args = f"--model_name_or_path {model_name_or_path} --data_path_train {data_path_train} --data_path_valid {data_path_valid} --output_path {output_path} --train_mode {train_mode} --message_mode {message_mode} --num_rounds {num_rounds}"
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
            peer_read_timeout=600,
            task_wait_timeout=None,            
            launch_timeout=None,               
            heartbeat_timeout=300.0,           
            last_result_transfer_timeout=900.0
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
        # TODO: this code is problematic, if i keep it i get AttributeError: 'dict' object has no attribute '__module__'. Did you mean: '__reduce__'?
        # if i remove the code, we can run, but the client timeout is not set and i keep hitting timeout errors (no quantization)
        # with quantization, no errors with timeout, training happens normaly 
        # client_params = {"task_result_timeout": 300}
        # job.to_clients(client_params)

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
        default="meta-llama/llama-3.2-1b",
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
    return parser.parse_args()


if __name__ == "__main__":
    main()
