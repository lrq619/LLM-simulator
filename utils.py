from simulate import simulate
import vllm
from vllm import SamplingParams
from mock import DataMocking
from typing import Tuple, List, Dict
import numpy as np
import time

def run_simulator(model_name: str, cuda_device_name: str, prompt_length: int, response_length: int, bsz=1) -> Tuple[float, float]:
    latencys, alpha, beta, c = simulate(
        model_name=model_name, 
        cuda_device_name=cuda_device_name, 
        prompt_length=prompt_length, 
        response_length=response_length,
        bsz=bsz
    )

    prompt_phase_latency = latencys[0]
    token_phase_latency = sum(latencys[1:])

    return prompt_phase_latency, token_phase_latency

def run_vllm(llm: vllm.LLM, cuda_device_name: str, prompt_length: int, response_length: int, bsz=1) -> Tuple[float, float]:

    data_mock = DataMocking()
    prompt = data_mock.create_prompt(prompt_token_len=prompt_length)
    prompts = [prompt for _ in range(bsz)]

    sample_param = SamplingParams(temperature=0, top_p=0.95, max_tokens=response_length, min_tokens=response_length - 1)
    outputs = llm.generate(prompts, sample_param)

    prompt_phase_latencys = []
    token_phase_latencys = []
    for output in outputs:
        metrics = output.metrics
        prompt_phase_latency = metrics.first_token_time - metrics.first_scheduled_time
        token_phase_latency = metrics.finished_time - metrics.first_token_time

        prompt_phase_latencys.append(prompt_phase_latency)
        token_phase_latencys.append(token_phase_latency)

    mean_prompt_phase_latency = np.asarray(prompt_phase_latencys).mean()
    mean_token_phase_latency = np.asarray(token_phase_latencys).mean()

    return mean_prompt_phase_latency, mean_token_phase_latency

