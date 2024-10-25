import torch
from typing import List, Dict, Tuple
import os
import json
import argparse
from utils import SHORT_GPU_NAME_DEVICE_CUDA_NAME_MAP, PROJECT_ROOT_PATH

def get_gpu_info(cuda_device_name: str) -> Tuple[int, float, float]:
    # Read the JSON file for memory bandwidth information
    try:
        json_file_name = os.path.join(PROJECT_ROOT_PATH, "./data/gpu.json")
        if not os.path.exists(json_file_name):
            err_msg = f"File not found: {json_file_name}"
            raise Exception(err_msg)

        with open(json_file_name, 'r') as file:
            data = json.load(file)
    
        # Check if CUDA device memory bandwidth information is available and valid
        memory_bw = data[cuda_device_name]["memory_bw"]
        if memory_bw < 0:
            raise Exception(f"memory bw for {cuda_device_name} is: {memory_bw} < 0, invalid!")
        # Example: Assuming memory_bw_util is also needed and has to be fetched or calculated
        # Here we just assign a placeholder value since how to get it isn't specified
        memory_bw_util = data[cuda_device_name]["memory_bw_util"]  # assuming 80% utilization for example purposes
        memory_cap = data[cuda_device_name]["memory_cap"]
        return memory_cap, memory_bw, memory_bw_util
    except Exception as e:
        print(e)
        err_msg = f"Could not find gpu information for {cuda_device_name} in {json_file_name}, consider first running:\n python profile_gpu.py\n and filling the memory_bandwidth values by hand"
        raise Exception(err_msg)

def get_model_info(model_name: str) -> Tuple[int, int, float, float]:
    try:
        json_file_name = os.path.join(PROJECT_ROOT_PATH,"./data/model.json")
        if not os.path.exists(json_file_name):
            err_msg = f"File not found: {json_file_name}"
            raise Exception(err_msg)

        with open(json_file_name, 'r') as file:
            data = json.load(file)

        num_hidden_layers = data[model_name]["num_hidden_layers"]
        num_heads = data[model_name]["num_heads"]
        model_size_GB = data[model_name]["model_size_GB"]
        kvc_size_KB = data[model_name]["kvc_size_KB"]
        return num_hidden_layers, num_heads, model_size_GB, kvc_size_KB
    except Exception as e:
        print(e)
        raise Exception(f"Could not find model information for {model_name} in {json_file_name}, consider first running: \npython profile_model.py --model-name={model_name}\n and check {json_file_name}")

def get_ptps(model_name: str, cuda_device_name: str) -> float:
    json_file_name = os.path.join(PROJECT_ROOT_PATH,"./data/ptps.json")
    try:
        if not os.path.exists(json_file_name):
            err_msg = f"File not found: {json_file_name}"
            raise Exception(err_msg)

        with open(json_file_name, 'r') as file:
            data = json.load(file)
        ptps = data[model_name][cuda_device_name]
        return ptps
    except Exception as e:
        print(e)
        raise Exception(f"Could not find ptps for {model_name} and {cuda_device_name} in {json_file_name}, consider first running: \npython profile_prompt.py --model-name={model_name}\n and check {json_file_name}")

def simulate(model_name: str, cuda_device_name: str, prompt_length: int, response_length: int, bsz: int=1) -> List[float]:
    """Simulates LLM generation latency for given model, prompt length and response length
    Return Value:
        Returns a list of float, where the first element denotes the prompt latency, rest element denotes the genertion latency for each token in token phase
    """
    latencys = []

    try:
        # First get gpu information
        memory_cap, memory_bw, memory_bw_util = get_gpu_info(cuda_device_name=cuda_device_name) 
        practical_mem_bw = memory_bw * memory_bw_util / 100 # GB/s

        # Then get model information
        num_hidden_layers, num_heads, model_size_GB, kvc_size_KB = get_model_info(model_name=model_name)

        # Finally get ptps
        ptps = get_ptps(model_name=model_name, cuda_device_name=cuda_device_name)

    except Exception as e:
        print(e)
        exit(1)

    print(f"For gpu: {cuda_device_name}, practical_mem_bw: {practical_mem_bw:.1f} GB/s, for model: {model_name}, kvc_size: {kvc_size_KB} KB, ptps={ptps:.1f} tokens/s")

    # Prompt Phase Latency
    prompt_token_num = prompt_length * bsz
    prompt_phase_latency = prompt_token_num / ptps
    latencys.append(prompt_phase_latency)

    kvc_size_GB = kvc_size_KB / (1024**2)
    for i in range(response_length):
        token_phase_latency = (model_size_GB + bsz * (prompt_length + i) * kvc_size_GB) / (practical_mem_bw)
        latencys.append(token_phase_latency)

    # alpha, beta, c in README
    alpha = kvc_size_GB / (2*practical_mem_bw)
    beta = (prompt_length + 0.5) * kvc_size_GB / practical_mem_bw
    c = model_size_GB / practical_mem_bw

    return latencys, alpha, beta, c



if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description="Model that is going to be simulated")

    # Add the argument for model name
    parser.add_argument('--model-name', type=str, help="Specify the model name", required=True)
    parser.add_argument('--gpu-name', type=str, help="Specify the gpu name", required=True)
    parser.add_argument('--prompt-length', type=int, help="Number of tokens in prompt", default=1024)
    parser.add_argument('--response-length', type=int, help="Number of tokens in response", default=128)
    parser.add_argument('--detail', type=bool, help="Whether to demonstrate deatils", default=False)

    # Parse the arguments
    args = parser.parse_args()

    # Extract the model name from the command line arguments
    model_name: str = args.model_name
    gpu_name: str = args.gpu_name
    prompt_length: int = args.prompt_length
    response_length: int = args.response_length
    detail : bool = args.detail

    cuda_device_name = SHORT_GPU_NAME_DEVICE_CUDA_NAME_MAP.get(gpu_name)
    if cuda_device_name == None:
        print(f"gpu name: {gpu_name} not supported!")
        exit(-1)
    # # Ensure CUDA is available
    # if torch.cuda.is_available():
    #     cuda_device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    #     if detail:
    #         print(f"We're simulating under {cuda_device_name}")
    # else:
    #     print(f"CUDA device not available! Cannot get ptps!")
    #     exit(1)

    latencys, alpha, beta, c = simulate(
        model_name=model_name, 
        cuda_device_name=cuda_device_name, 
        prompt_length=prompt_length, 
        response_length=response_length
    )
    prompt_phase_latency = latencys[0]
    token_phase_latency = sum(latencys[1:])
    total_latency = prompt_phase_latency + token_phase_latency
    if detail: 
        print(f"model: {model_name} running on {cuda_device_name}, with prompt and response: {prompt_length}-{response_length}\n\tPrompt phase latency: {prompt_phase_latency:.2f}s\n\tToken phase latency: {token_phase_latency:.2f}s\n\tTotal latency: {total_latency:.2f}s\n\t alpha={alpha}, beta={beta}, c={c}, c/beta={c/beta:.1f}")
    print(f"latencys: {latencys}")

    
    