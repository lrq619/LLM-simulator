import json
import os
SHORT_GPU_NAME_DEVICE_CUDA_NAME_MAP = {
    "a100": "NVIDIA A100 80GB PCIe",
    "A100": "NVIDIA A100 80GB PCIe",
    "a5000": "NVIDIA RTX A5000",
    "A5000": "NVIDIA RTX A5000",
    "a6000": "NVIDIA RTX A6000",
    "A6000": "NVIDIA RTX A6000",
}

def get_project_root_path():
    # Get the absolute path to the directory containing this file (utils.py)
    return os.path.abspath(os.path.dirname(__file__))

PROJECT_ROOT_PATH = get_project_root_path()

def get_model_info(model_name: str):
    with open(os.path.join(PROJECT_ROOT_PATH, "data/model.json"), 'r') as f:
        model_info_dict = json.load(f)
        return model_info_dict.get(model_name, None)

def get_gpu_info(cuda_device_name: str):
    with open(os.path.join(PROJECT_ROOT_PATH,"data/gpu.json"), 'r') as f:
        gpu_info_dict = json.load(f)
        return gpu_info_dict.get(cuda_device_name, None)


