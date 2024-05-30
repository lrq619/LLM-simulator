import json
import os
import torch
import math

def main():
    # File path
    json_file_path = './data/gpu.json'

    # Ensure the directory exists
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)

    # Try to read the existing JSON data
    try:
        with open(json_file_path, 'r') as file:
            json_obj = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        # If file doesn't exist or is empty, initialize a new JSON object
        json_obj = {}

    # Check CUDA availability and get device info
    if torch.cuda.is_available():
        cuda_device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        memory_capacity = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        # Convert memory capacity from bytes to gigabytes
        memory_capacity_gb = math.ceil(memory_capacity / (1024**3))
        
        # Update JSON object
        json_obj[cuda_device_name] = {
            "memory_cap": memory_capacity_gb,
            "memory_bw": -1,
            "memory_bw_util": 60,
        }

        # Print message about profiling
        print(f"{cuda_device_name} has been profiled, please provide its memory bandwidth(GB/s) in {json_file_path}")
    else:
        print("CUDA device not available!")
        exit(1)

    # Write updated JSON object back to the file
    with open(json_file_path, 'w') as file:
        json.dump(json_obj, file, indent=4)

if __name__ == "__main__":
    main()