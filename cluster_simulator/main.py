import json
import argparse
from cluster_simulator import simulate
from cluster_simulator.trace import process_trace


def main():
    parser = argparse.ArgumentParser(description="cluster level simulator")
    parser.add_argument("--config", type=str, default="./cluster_simulator/config.json")

    args = parser.parse_args()
    config_file = args.config
    with open(config_file, 'r') as f:
        config = json.load(f)

    workloads = config["workloads"]
    for workload_config in workloads:
        processed_trace_file_path = process_trace(
            trace_path=workload_config["trace_path"],
            sampling_rate=workload_config["sampling_rate"],
            input_length_scale=workload_config["input_length_scale"],
            output_length_scale=workload_config["output_length_scale"]
        )
        print(f"Process trace finished! Going to use {processed_trace_file_path} as the trace for model: {workload_config['model']}")

if __name__ == '__main__':
    main()