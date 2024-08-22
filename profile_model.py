import argparse
import os
from transformers import AutoConfig
import json
import re

def profile_model(model_name):
    # Path to JSON file
    json_file_path = './data/model.json'

    # Ensure the directory exists
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)

    # Load or initialize JSON data
    try:
        with open(json_file_path, 'r') as file:
            json_obj = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        json_obj = {}

    # Load the model configuration from Hugging Face
    config = AutoConfig.from_pretrained(model_name)

    # Extract the required attributes
    num_hidden_layers = config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else None
    if hasattr(config, "num_key_values_heads"):
        num_heads = config.num_key_values_heads
    else:
        num_heads = config.num_attention_heads 
    # Regular expression pattern to find the number
    pattern = r"/([^\s/]+)-(\d+\.\d+|\d+)(b|m|B)"
    match = re.search(pattern, model_name)
    num_params = float(match.group(2))
    model_size_GB = (num_params * 2)  # Assuming parameters are 32-bit floats, 2 bytes each

    # Calculate kvc_size_KB
    d_head = (config.hidden_size // num_heads) if num_heads else None
    precision_bytes = 2  # Assuming we're working with 16-bit precision for this calculation
    kvc_size_KB = (2 * num_hidden_layers * num_heads * d_head * precision_bytes) / 1024 if num_heads and d_head else None

    # Update JSON object
    json_obj[model_name] = {
        "num_hidden_layers": num_hidden_layers,
        "num_heads": num_heads,
        "model_size_GB": model_size_GB,
        "kvc_size_KB": kvc_size_KB
    }

    # Write back to JSON
    with open(json_file_path, 'w') as file:
        json.dump(json_obj, file, indent=4)

    print(f"Profiled {model_name} and updated {json_file_path}")

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description="Model that is going to be profiled")

    # Add the argument for model name
    parser.add_argument('--model-name', type=str, help="Specify the model name", required=True)

    # Parse the arguments
    args = parser.parse_args()

    # Extract the model name from the command line arguments
    model_name: str = args.model_name

    profile_model(model_name)