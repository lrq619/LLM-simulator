import os
import subprocess
from cluster_simulator import PROJECT_DIR

def process_trace(trace_path: str, sampling_rate, input_length_scale, output_length_scale) -> str:
    # check if the processed trace already under processed_trace
    trace = os.path.splitext(os.path.basename(trace_path))[0]
    process_trace_file_name = f"{trace}-{sampling_rate}-{input_length_scale}-{output_length_scale}.json"
    processed_trace_file_path = os.path.join(PROJECT_DIR, "trace_processed", process_trace_file_name)
    if os.path.exists(processed_trace_file_path):
        print(f"{processed_trace_file_path} already exists, going to reuse it.")
        pass
    else:
        # sample the file
        sampler_bin_path = os.path.join(PROJECT_DIR, "sampler/sampler") 
        dataset_path = trace_path

        cmd = f"{sampler_bin_path} -dataset_path {dataset_path} -sampling_rate {sampling_rate} -result_path {processed_trace_file_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True) 
        print(result.stdout)
    
    return processed_trace_file_path

