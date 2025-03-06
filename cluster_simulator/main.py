import json
import argparse
import matplotlib.pyplot as plt
from cluster_simulator import simulate, PROJECT_DIR
from cluster_simulator.trace import process_trace
from cluster_simulator.time_series import convert_processed_trace_to_concurrency_series, convert_concurrency_to_chunk_number_series,extract_alloc_free_events, TimeSeriesFunction
from cluster_simulator.hardware import ClusterManager, NodeInfo, GPUInfo
MAX_CHUNK_SIZE = 4


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

    idle_gpu_number_series, cont_gpu_number_series = cluster_manager.replay(gpu_operations, max_chunk_size=MAX_CHUNK_SIZE)
    plot_idle_and_cont_gpu_numbers(idle_gpu_number_series, cont_gpu_number_series).savefig(f"{PROJECT_DIR}/results/gpu_number.png")

    # idle_gpu_number_series.plot().savefig(f"{PROJECT_DIR}/results/idle_gpu_number.png")
    # cont_gpu_number_series.plot().savefig(f"{PROJECT_DIR}/results/cont_gpu_number.png")

def plot_idle_and_cont_gpu_numbers(idle_gpu_number_series: TimeSeriesFunction, cont_gpu_number_series: TimeSeriesFunction) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(cont_gpu_number_series.timestamps, cont_gpu_number_series.values, color="blue", label="Continous Idle GPUs", zorder=3)
        ax.plot(idle_gpu_number_series.timestamps, idle_gpu_number_series.values, color="red", label="Total Idle GPUs", zorder=3)

        ax.set_xlabel("Timestamps")
        ax.set_ylabel("GPU Number")
        ax.grid(True)
        ax.legend()

        return fig  # Return the figure instance
        

if __name__ == '__main__':
    main()