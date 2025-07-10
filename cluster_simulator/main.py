import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from cluster_simulator import simulate, PROJECT_DIR
from cluster_simulator.trace import process_trace
from cluster_simulator.time_series import convert_processed_trace_to_concurrency_series, convert_concurrency_to_chunk_number_series,extract_alloc_free_events, convert_processed_trace_to_utilization_series, TimeSeriesFunction, EventTimestamp
from cluster_simulator.hardware import ClusterManager, NodeInfo, GPUInfo
from cluster_simulator.autoscaler import save_timeseries_to_json, normalize_ts
MAX_CHUNK_SIZE = 4
from typing import List
os.makedirs(f"{PROJECT_DIR}/results", exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="cluster level simulator")
    parser.add_argument("--config", type=str, default="./cluster_simulator/config.json")
    parser.add_argument("--autoscale", type=str, default="kpa")

    args = parser.parse_args()
    config_file = args.config
    autoscale = args.autoscale
    with open(config_file, 'r') as f:
        config = json.load(f)
    workloads = config["workloads"]
    utilization_target = config["utilization_target"] / 100
    
    with open(f"{PROJECT_DIR}/data/gpu.json", 'r') as f:
        gpu_info = json.load(f)
        
    gpu_name = config["gpu_name"]
    gpu_number = config["gpu_number"]
    gpu_info = gpu_info[gpu_name]
    gpu_memory_cap = gpu_info["memory_cap"]
    
    # setup the cluster manager
    node_number = config["node_number"]
    gpu_number = config["gpu_number"]
    cluster_manager = ClusterManager(node_number=node_number, gpu_number=gpu_number, workload_number=len(workloads))


    # iterate over the workloads
    gpu_operations = []
    for workload_id, workload_config in enumerate(workloads):
        model_name = workload_config["model"]
        with open(f"{PROJECT_DIR}/data/model.json", 'r') as f:
            model_info = json.load(f)
        model_info = model_info[model_name]
        model_size_GB = model_info["model_size_GB"]
        model_utilization = model_size_GB / gpu_memory_cap
        left_space_utilization = utilization_target - model_utilization
        print(f"Left_space_utilization: {left_space_utilization}")
        chunk_size = workload_config.get("tp_level", 1)
        print(f"Processing trace for workload {workload_id} with chunk size {chunk_size}")
        processed_trace_file_path = process_trace(
            trace_path=workload_config["trace_path"],
            sampling_rate=workload_config["sampling_rate"],
            input_length_scale=workload_config["input_length_scale"],
            output_length_scale=workload_config["output_length_scale"]
        )
        print(f"Process trace finished! Going to use {processed_trace_file_path} as the trace for model: {workload_config['model']}")
        # Input the processed trace file, output the timeseries concurrency vs. time
        if autoscale == "default" or autoscale == "kpa":
            concurrency_series = convert_processed_trace_to_concurrency_series(processed_trace_file_path, workload_config['model'],config["gpu_name"])
            chunk_number_series = convert_concurrency_to_chunk_number_series(concurrency_series, autoscale, workload_config['target'])
            save_timeseries_to_json(concurrency_series, f"{PROJECT_DIR}/results/concurrency_{workload_id}.json")
            plt.plot(concurrency_series.timestamps, concurrency_series.values)
            plt.ylabel(f"Concurrency")
            plt.xlabel(f"Timestamps")
            plt.savefig(f"{PROJECT_DIR}/results/concurrency_{workload_id}.png")
        elif autoscale == "hpa" or autoscale == "apa":
            concurrency_series = convert_processed_trace_to_utilization_series(processed_trace_file_path, workload_config['model'],config["gpu_name"])
            chunk_number_series = convert_concurrency_to_chunk_number_series(concurrency_series, autoscale, left_space_utilization)
            # all utilization should be added 
            save_timeseries_to_json(concurrency_series, f"{PROJECT_DIR}/results/utilization_{workload_id}.json")
            plt.plot(concurrency_series.timestamps, concurrency_series.values)
            plt.ylabel(f"Utilization")
            plt.xlabel(f"Timestamps")
            plt.savefig(f"{PROJECT_DIR}/results/utilization_{workload_id}.png")
        
        with open(f"{PROJECT_DIR}/results/chunk_number_{workload_id}.json", "w") as f:
            json.dump({
                "timestamps": chunk_number_series.timestamps.tolist(), 
                "values": chunk_number_series.values.tolist()
            }, f)
        fig = chunk_number_series.plot()
        ax = fig.gca()
        ax.set_ylabel(f"GPU Chunk Number")
        ax.set_xlabel(f"Timestamps")
        fig.savefig(f"{PROJECT_DIR}/results/chunk_number_{workload_id}.png")
        # An event list recording each free/alloc GPU
        operations = extract_alloc_free_events(chunk_number_series, workload_id, chunk_size, node_number, gpu_number)
        gpu_operations.extend(operations)

    print("length of gpu_operations: ", gpu_operations[0].event)
    idle_gpu_number_series, cont_gpu_number_series, alloc_events = cluster_manager.replay(gpu_operations, max_chunk_size=MAX_CHUNK_SIZE)
    process_alloc_events(alloc_events)
    print("Process_alloc_events finished!")
    # keep 1 timestamp per second
    normalized_idle_series = normalize_ts(idle_gpu_number_series)
    alloc_success_number = node_number * gpu_number - normalized_idle_series.values
    alloc_success_number_series = TimeSeriesFunction(normalized_idle_series.timestamps, alloc_success_number)
    save_timeseries_to_json(alloc_success_number_series, f"{PROJECT_DIR}/results/alloc_success_number.json")
    print(f"Avg running #GPU: {np.mean(alloc_success_number)}")
    fig_alloc, ax_alloc = plt.subplots()
    ax_alloc.plot(alloc_success_number_series.timestamps, alloc_success_number_series.values)
    ax_alloc.set_ylabel(f"Running GPU Number")
    ax_alloc.set_xlabel(f"Timestamps")
    fig_alloc.savefig(f"{PROJECT_DIR}/results/alloc_success_number.png")
    print("Alloc succeed!")
    # Get the failure alloc series, when the try_alloc_gpu_number is larger than the idle_gpu_number
    failure_alloc_series = get_failure_alloc_series(alloc_events)
    
    plot_fragmentation_gpu_number(failure_alloc_series).savefig(f"{PROJECT_DIR}/results/fragmentation_gpu_number.png")
    plot_idle_and_cont_gpu_numbers(idle_gpu_number_series, cont_gpu_number_series).savefig(f"{PROJECT_DIR}/results/gpu_number.png")
    # sampled_cont_gpu_timestamps, sampled_cont_gpu_number = cont_gpu_number_series.sample()
    plot_idle_and_cont_gpu_numbers_cdf(normalized_idle_series.values, cont_gpu_number_series.values).savefig(f"{PROJECT_DIR}/results/cdf.png")

def process_alloc_events(alloc_events: List[EventTimestamp]):
    data_dict = {}
    for event_ts in alloc_events:
        event = event_ts.event
        workload_id = event["workload_id"]
        if workload_id not in data_dict.keys():
             data_dict[workload_id] = []
        data_dict[workload_id].append({
             "success": event["success"],
             "try_alloc_gpu_number": event["chunk_size"] * event["delta_chunk_number"],
             "idle_gpu_number": event["idle_gpu_number"]
        })

    print(f"----------------------------------------------------------------------------------------------------------------------------------------------------")
    for workload_id, data_list in data_dict.items():
        total_alloc_number = len(data_list)
        failure_alloc_number = 0
        fragmentation_alloc_number = 0
        for data in data_list:
            if data["success"] == False:
                failure_alloc_number += 1
                if data["try_alloc_gpu_number"] <= data["idle_gpu_number"]:
                    fragmentation_alloc_number += 1

        print(f"For workload: {workload_id}, there are totally {total_alloc_number} allocs, where {failure_alloc_number} are failed. In these failed allocs, {fragmentation_alloc_number} are failed due to external fragmentation.")
    print(f"----------------------------------------------------------------------------------------------------------------------------------------------------")
    
def get_failure_alloc_series(alloc_events: List[EventTimestamp]) -> TimeSeriesFunction:
    failure_alloc_dict = {}
    for event in alloc_events:
        if event.ts not in failure_alloc_dict.keys():
            if event.event["success"] == False:
                failure_alloc_dict[event.ts] = event.event["idle_gpu_number"]
            else:
                failure_alloc_dict[event.ts] = 0
    failure_alloc_dict_sorted = sorted(failure_alloc_dict.items(), key=lambda x: x[0])
    timestamps = [item[0] for item in failure_alloc_dict_sorted]
    values = [item[1] for item in failure_alloc_dict_sorted]
    failure_alloc_series = TimeSeriesFunction(timestamps, values)
    return failure_alloc_series


def plot_idle_and_cont_gpu_numbers(idle_gpu_number_series: TimeSeriesFunction, cont_gpu_number_series: TimeSeriesFunction) -> plt.Figure:
        fig, axs = plt.subplots(2,1,figsize=(8, 4), sharex=True)

        axs[0].plot(idle_gpu_number_series.timestamps, idle_gpu_number_series.values, color="red", label="Total Idle GPUs", zorder=3)
        axs[1].plot(cont_gpu_number_series.timestamps, cont_gpu_number_series.values, color="blue", label="Continous Idle GPUs", zorder=3)

        axs[0].set_xlabel("Timestamps")
        axs[0].set_ylabel("Idle #GPU")
        axs[1].set_ylabel("Continous #GPU")
        # ax.grid(True)
        # ax.legend()

        return fig  # Return the figure instance

def plot_idle_and_cont_gpu_numbers_cdf(sampled_idle_gpu_number, sampled_cont_gpu_number) -> plt.Figure:
    print(f"Avg idle #GPU: {np.mean(np.mean(sampled_idle_gpu_number))}, Avg continous #GPU: {np.mean(sampled_cont_gpu_number)}")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(np.sort(sampled_idle_gpu_number), np.linspace(0, 1, len(sampled_idle_gpu_number), endpoint=False), label=f"Idle GPU number", color="red")
    ax.plot(np.sort(sampled_cont_gpu_number), np.linspace(0, 1, len(sampled_cont_gpu_number), endpoint=False), label=f"Continous GPU number(chunk=4)", color="blue")
    ax.set_xlabel(f"GPU number")
    ax.set_ylabel(f"CDF")
    ax.legend()
    ax.grid(True)
    return fig

def plot_fragmentation_gpu_number(failure_alloc_series: TimeSeriesFunction) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(failure_alloc_series.timestamps, failure_alloc_series.values, color="red", label="Fragmentation GPU number", zorder=3)
    ax.set_xlabel(f"Timestamps")
    ax.set_ylabel(f"Fragmentation GPU number")
    ax.legend()
    return fig

if __name__ == '__main__':
    main()