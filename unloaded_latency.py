import argparse
import csv
import json
from simulate import simulate
from utils import SHORT_GPU_NAME_DEVICE_CUDA_NAME_MAP

def read_trace_file(trace_file):
    data = []
    with open(trace_file, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            # Collecting ContextTokens and GeneratedTokens
            context_tokens = int(row["ContextTokens"])
            generated_tokens = int(row["GeneratedTokens"])
            data.append([context_tokens, generated_tokens])
    return data

def main():
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Process trace file for unloaded latency.")
    parser.add_argument('--trace_file', type=str, required=True, help='Path to the trace CSV file')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--gpu_name', type=str, required=True, help='Name of the GPU')
    parser.add_argument('--output_file_name', type=str, required=False, help='Name of the output file')

    # Parse arguments
    args = parser.parse_args()

    # Read trace file
    token_data = read_trace_file(args.trace_file)

    cuda_device_name = SHORT_GPU_NAME_DEVICE_CUDA_NAME_MAP[args.gpu_name]

    e2e_latencys = []
    for prompt_length, response_length in token_data:
        latencys, _, _, _ = simulate(
            args.model_name, cuda_device_name, prompt_length, response_length
        )
        e2e_latency = sum(latencys)
        e2e_latencys.append(e2e_latency)

    model_suffix = args.model_name.split('/')[-1]
    if args.output_file_name:
        output_file_name = args.output_file_name
    else:
        input_name = args.trace_file.removesuffix('.csv')
        output_file_name = f"{input_name}_slowdown-{model_suffix}.json"

    outputs = []
    with open(output_file_name, 'w+') as f:
        for i, (prompt_length, response_length) in enumerate(token_data):
            latency = e2e_latencys[i]
            outputs.append(
                {
                    "ContextTokens": prompt_length,
                    "GeneratedTokens": response_length,
                    "SlowdownBase": latency,
                }
            )
        json.dump(outputs, f)
            



if __name__ == "__main__":
    main()
