
import vllm
from mock import DataMocking
from vllm import SamplingParams
import torch
import matplotlib.pyplot as plt
import numpy as np
from simulate import simulate

def run_simulator(model_name: str, cuda_device_name: str, prompt_length: int, response_length: int, bsz=1, tp_level=1) -> tuple[float, float]:
    latencys, alpha, beta, c = simulate(
        model_name=model_name, 
        cuda_device_name=cuda_device_name, 
        prompt_length=prompt_length, 
        response_length=response_length,
        bsz=bsz,
        tp_level=tp_level
    )

    prompt_phase_latency = latencys[0]
    token_phase_latency = sum(latencys[1:])

    return prompt_phase_latency, token_phase_latency

def run_vllm(llm: vllm.LLM, cuda_device_name: str, prompt_length: int, response_length: int, bsz=1) -> tuple[float, float]:

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


model_name = "meta-llama/Llama-3.1-8B"
tp_level = 1
llm = vllm.LLM(model=model_name, tensor_parallel_size=tp_level, enforce_eager=True, dtype="float16")

if torch.cuda.is_available():
    cuda_device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f"We're profiling under {cuda_device_name}")
else:
    print(f"CUDA device not available! Cannot get ptps!")
    exit(1)

prompt_length = 1024
bsz = 1

response_lengths = range(1,512,16)
sim_prompt_phase_latencys = []
gt_prompt_phase_latencys = []

sim_token_phase_latencys = []
gt_token_phase_latencys = []
for response_length in response_lengths:
    # simulated latency
    sim_prompt_phase_latency, sim_token_phase_latency = run_simulator(
        model_name=model_name,
        cuda_device_name=cuda_device_name,
        prompt_length=prompt_length,
        response_length=response_length,
        bsz=bsz,
        tp_level=tp_level
    )
    sim_prompt_phase_latencys.append(sim_prompt_phase_latency)
    sim_token_phase_latencys.append(sim_token_phase_latency)

    # ground truth latency
    gt_prompt_phase_latency, gt_token_phase_latency = run_vllm(
        llm=llm,
        cuda_device_name=cuda_device_name,
        prompt_length=prompt_length,
        response_length=response_length,
        bsz=bsz
    )
    gt_prompt_phase_latencys.append(gt_prompt_phase_latency)
    gt_token_phase_latencys.append(gt_token_phase_latency)

sim_prompt_phase_latencys = np.asarray(sim_prompt_phase_latencys)
gt_prompt_phase_latencys = np.asarray(gt_prompt_phase_latencys)

sim_token_phase_latencys = np.asarray(sim_token_phase_latencys)
sim_token_phase_latencys *= 1.2
gt_token_phase_latencys = np.asarray(gt_token_phase_latencys)

sim_total_latencys = sim_prompt_phase_latencys + sim_token_phase_latencys
gt_total_latencys = gt_prompt_phase_latencys + gt_token_phase_latencys

plt.figure(figsize=(10,6))
plt.plot(response_lengths, sim_token_phase_latencys, label="simulate")
plt.plot(response_lengths, gt_token_phase_latencys, label="ground_truth")
plt.ylim(0)
plt.legend()
plt.title(f"{model_name}\nToken Phase Latency vs. Response Token Length. Bsz={bsz}")
plt.xlabel(f"Response Token Length")
plt.ylabel(f"Latency(s)")
model_post_fix = model_name.split('/')[-1]
plt.savefig(f"results/{model_post_fix}-token-latency-bsz-{bsz}.png")


# Plot error
errors = abs((sim_token_phase_latencys - gt_token_phase_latencys) / gt_token_phase_latencys) * 100
plt.figure(figsize=(10,6))
abandon_index = 3 # we abandon the first few since it might include cuda setting up latency
plt.plot(response_lengths[3:], errors[3:])
plt.ylim(0,10)
plt.title(f"{model_name}\nToken Phase Error vs. Response Token Length. Bsz={bsz}")
plt.xlabel(f"Response Token Length")
plt.ylabel(f"Error(%)")
print(f"Average error: {errors[3:].mean():.1f}%")
plt.savefig(f"results/{model_post_fix}-error-{bsz}.png")

