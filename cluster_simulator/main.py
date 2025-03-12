import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from cluster_simulator import simulate, PROJECT_DIR
from cluster_simulator.trace import process_trace
from cluster_simulator.time_series import convert_processed_trace_to_concurrency_series, convert_concurrency_to_chunk_number_series,extract_alloc_free_events, TimeSeriesFunction, EventTimestamp
from cluster_simulator.hardware import ClusterManager, NodeInfo, GPUInfo
MAX_CHUNK_SIZE = 4
from typing import List


def main():
    parser = argparse.ArgumentParser(description="cluster level simulator")
    parser.add_argument("--config", type=str, default="./cluster_simulator/config.json")

    args = parser.parse_args()
    config_file = args.config
    with open(config_file, 'r') as f:
        config = json.load(f)

    workloads = config["workloads"]
    # setup the cluster manager
    node_number = config["node_number"]
    gpu_number = config["gpu_number"]
    cluster_manager = ClusterManager(node_number=node_number, gpu_number=gpu_number, workload_number=len(workloads))

    # iterate over the workloads
    gpu_operations = []
    for workload_id, workload_config in enumerate(workloads):
        chunk_size = workload_config.get("tp_level", 1)
        processed_trace_file_path = process_trace(
            trace_path=workload_config["trace_path"],
            sampling_rate=workload_config["sampling_rate"],
            input_length_scale=workload_config["input_length_scale"],
            output_length_scale=workload_config["output_length_scale"]
        )
        print(f"Process trace finished! Going to use {processed_trace_file_path} as the trace for model: {workload_config['model']}")
        # Input the processed trace file, output the timeseries concurrency vs. time
        concurrency_series = convert_processed_trace_to_concurrency_series(processed_trace_file_path, config["gpu_name"])
        fig = concurrency_series.plot()
        ax = fig.gca() 
        ax.set_ylabel(f"Concurrency")
        ax.set_xlabel(f"Timestamps")
        fig.savefig(f"{PROJECT_DIR}/results/concurrency_{workload_id}.png")
        chunk_number_series = convert_concurrency_to_chunk_number_series(concurrency_series, workload_config['target'])
        fig = chunk_number_series.plot()
        ax = fig.gca()
        ax.set_ylabel(f"GPU Chunk Number")
        ax.set_xlabel(f"Timestamps")
        fig.savefig(f"{PROJECT_DIR}/results/chunk_number_{workload_id}.png")
        # An event list recording each free/alloc GPU
        operations = extract_alloc_free_events(chunk_number_series, workload_id, chunk_size)
        gpu_operations.extend(operations)

    idle_gpu_number_series, cont_gpu_number_series, alloc_events = cluster_manager.replay(gpu_operations, max_chunk_size=MAX_CHUNK_SIZE)
    process_alloc_events(alloc_events)
    plot_idle_and_cont_gpu_numbers(idle_gpu_number_series, cont_gpu_number_series).savefig(f"{PROJECT_DIR}/results/gpu_number.png")
    sampled_idle_gpu_timestamps, sampled_idle_gpu_number = idle_gpu_number_series.sample()
    sampled_cont_gpu_timestamps, sampled_cont_gpu_number = cont_gpu_number_series.sample()
    plot_idle_and_cont_gpu_numbers_cdf(sampled_idle_gpu_number, sampled_cont_gpu_number).savefig(f"{PROJECT_DIR}/results/cdf.png")

    # idle_gpu_number_series.plot().savefig(f"{PROJECT_DIR}/results/idle_gpu_number.png")
    # cont_gpu_number_series.plot().savefig(f"{PROJECT_DIR}/results/cont_gpu_number.png")

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
                if data["try_alloc_gpu_number"] < data["idle_gpu_number"]:
                    fragmentation_alloc_number += 1

        print(f"For workload: {workload_id}, there are totally {total_alloc_number} allocs, where {failure_alloc_number} are failed. In these failed allocs, {fragmentation_alloc_number} are failed due to external fragmentation.")
    print(f"----------------------------------------------------------------------------------------------------------------------------------------------------")
    

def plot_idle_and_cont_gpu_numbers(idle_gpu_number_series: TimeSeriesFunction, cont_gpu_number_series: TimeSeriesFunction) -> plt.Figure:
        fig, axs = plt.subplots(2,1,figsize=(8, 4), sharex=True)

        axs[0].plot(idle_gpu_number_series.timestamps, idle_gpu_number_series.values, color="red", label="Total Idle GPUs", zorder=3)
        axs[1].plot(cont_gpu_number_series.timestamps, cont_gpu_number_series.values, color="blue", label="Continous Idle GPUs", zorder=3)

        axs[0].set_xlabel("Timestamps")
        axs[0].set_ylabel("Total #GPU")
        axs[1].set_ylabel("Continous #GPU")
        # ax.grid(True)
        # ax.legend()

        return fig  # Return the figure instance

def plot_idle_and_cont_gpu_numbers_cdf(sampled_idle_gpu_number, sampled_cont_gpu_number) -> plt.Figure:
    print(f"Avg total #GPU: {np.mean(np.mean(sampled_idle_gpu_number))}, Avg continous #GPU: {np.mean(sampled_cont_gpu_number)}")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(np.sort(sampled_idle_gpu_number), np.linspace(0, 1, len(sampled_idle_gpu_number), endpoint=False), label=f"Total GPU number", color="red")
    ax.plot(np.sort(sampled_cont_gpu_number), np.linspace(0, 1, len(sampled_cont_gpu_number), endpoint=False), label=f"Continous GPU number(chunk=4)", color="blue")
    ax.set_xlabel(f"GPU number")
    ax.set_ylabel(f"CDF")
    ax.legend()
    ax.grid(True)
    return fig
     
        

if __name__ == '__main__':
    main()