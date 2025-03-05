import vllm
from vllm import SamplingParams
from mock import DataMocking
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import torch
import pandas as pd
import argparse
import json
import os

def profile_prompt_phase(llm, prompt_token_num: int, bsz: int=1):
    prompt_length = prompt_token_num // bsz
    response_length = 1

    data_mock = DataMocking()
    prompt = data_mock.create_prompt(prompt_token_len=prompt_length)
    prompts = [prompt for _ in range(bsz)]

    sample_param = SamplingParams(temperature=0, top_p=0.95, max_tokens=response_length, min_tokens=response_length - 1)

    outputs = llm.generate(prompts, sample_param)
    prompt_phase_latencys = []
    for output in outputs:
        metrics = output.metrics
        prompt_phase_latency = metrics.first_token_time - metrics.first_scheduled_time
        prompt_phase_latencys.append(prompt_phase_latency)

    mean_prompt_phase_latency = np.asarray(prompt_phase_latencys).mean()
    

    return mean_prompt_phase_latency

def verify_latency_prompt_num(model_name: str):
    """This experiment verifies that prompt phase latency is only related to the total number of tokens processing, not related to the bsz
    """
    bszs = [1,2,4,8,16]
    prompt_token_num = 1024
    latencys = []
    llm = vllm.LLM(model=model_name)
    for bsz in bszs:
        prompt_phase_latency = profile_prompt_phase(
            llm=llm, 
            prompt_token_num=prompt_token_num, 
            bsz=bsz
        )
        latencys.append(prompt_phase_latency)

    bszs_str = [str(bsz) for bsz in bszs]
    plt.bar(bszs_str, latencys)
    plt.xlabel(f"Batch Size")
    plt.ylabel(f"Prompt Phase Latency(s)")
    plt.ylim(0)
    plt.title(f"{model_name}\n Prompt Phase Latency with {prompt_token_num} Tokens Divided into Batches")
    fig_file_name = f"results/prompt-latency-{prompt_token_num}tokens.png"
    plt.savefig(fig_file_name)

def get_latency_vs_prompt_num(model_name: str, max_prompt_num: int) -> float:
    prompt_token_nums = range(16, max_prompt_num, 16)
    latencys = []
    llm = vllm.LLM(model=model_name)
    for prompt_token_num in prompt_token_nums:
        prompt_phase_latency = profile_prompt_phase(
            llm=llm, 
            prompt_token_num=prompt_token_num, 
            bsz=1
        )
        latencys.append(prompt_phase_latency)

    slope, intercept, r_value, p_value, std_err = stats.linregress(latencys, prompt_token_nums)
    plt.plot(prompt_token_nums, latencys)
    plt.xlabel(f"Prompt Token Number")
    plt.ylabel(f"Prompt Phase Latency(s)")
    plt.ylim(0)
    plt.title(f"{model_name}\n Prompt Phase Latency with Different Prompt Token Number")
    model_post_fix = model_name.split('/')[-1]
    fig_file_name = f"results/{model_post_fix}-prompt-latencys.png"
    plt.savefig(fig_file_name)

    return slope

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description="Process the model name.")

    # Add the argument for model name
    parser.add_argument('--model-name', type=str, help="Specify the model name", required=True)
    parser.add_argument('--max-prompt-num', type=int, help="Specify the maximal number of prompt tokens that is going to be profiled", required=False, default=2048)

    # Parse the arguments
    args = parser.parse_args()

    # Extract the model name from the command line arguments
    model_name: str = args.model_name
    max_prompt_num: int = args.max_prompt_num

    # verify_latency_prompt_num(model_name=model_name)
    # Get the prompt token process speed(ptps)(token/s) for this model
    model_post_fix = model_name.split('/')[-1]
    ptps = get_latency_vs_prompt_num(model_name=model_name, max_prompt_num=max_prompt_num)


    # Ensure CUDA is available
    if torch.cuda.is_available():
        cuda_device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"We're profiling under {cuda_device_name}")
    else:
        print(f"CUDA device not available! Cannot get ptps!")
        exit(1)
    
    # Filename and path
    data_json_filename = './data/ptps.json'

    # Ensure the directory exists
    os.makedirs(os.path.dirname(data_json_filename), exist_ok=True)

    # Try to open the JSON file, or create it if it doesn't exist
    try:
        with open(data_json_filename, 'r') as file:
            # Try to load the JSON object
            json_obj = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        # If the file doesn't exist or is empty, initialize a new JSON object
        json_obj = {}

    # Update the JSON object with the new data
    if model_name not in json_obj:
        json_obj[model_name] = {}
    json_obj[model_name][cuda_device_name] = ptps

    # Write the JSON object back to the file
    with open(data_json_filename, 'w') as file:
        json.dump(json_obj, file, indent=4)

    print(f"Data updated in {data_json_filename}")