import json
import pandas as pd
from piecewise_func import PiecewiseLinear
from simulate import simulate
from utils import PROJECT_ROOT_PATH
import os


def get_latency(_model_name: str, _cuda_device_name: str, _prompt_length: int, _response_length: int):
    latencies,_ ,_ ,_ = simulate(
        model_name = _model_name,
        cuda_device_name = _cuda_device_name,
        prompt_length = _prompt_length,
        response_length = _response_length 
        )
    prompt_latency = latencies[0]
    response_latency = sum(latencies[1:])
    total_latency = prompt_latency + response_latency
    
    return prompt_latency, response_latency, total_latency

def get_kvc_size(_model_name: str):
    with open(os.path.join(PROJECT_ROOT_PATH,'data/model.json'),'r') as file:
        data = json.load(file)
        
    model_info = data[_model_name]
    kvc_in_kb = model_info['kvc_size_KB']
    kvc_in_gb = kvc_in_kb / (1024 ** 2)
    return kvc_in_gb

def load_datasets(_dataset_content: str):
    data = pd.read_csv(os.path.join(PROJECT_ROOT_PATH, f'datasets/AzureLLMInferenceTrace_{_dataset_content}.csv'), skiprows = 1,
                       header = None, names=['TIMESTAMP', 'ContextTokens', 'GeneratedTokens'])
    data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])
    initial_time = data['TIMESTAMP'].iloc[0]
    data['TimeDelta'] = (data['TIMESTAMP'] - initial_time).dt.total_seconds()
    #data['TimeDelta'] = data['TimeDelta'].astype(int)
    result_df = pd.DataFrame({
        'TIMESTAMP': data['TimeDelta'],
        'ContextTokens': data['ContextTokens'],
        'GeneratedTokens': data['GeneratedTokens']
        })
    
    return result_df
    