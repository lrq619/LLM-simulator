import os
import yaml
import json
import argparse
import numpy as np
from typing import List
from concurrent.futures import ThreadPoolExecutor
from cluster_simulator.common import TimeSeriesFunction, EventTimestamp


processed_trace_file_path = "../trace_processed/sample_0_all_input1_output1_ALL-30-1-1.json"
parser = argparse.ArgumentParser()
parser.add_argument("--autoscale", type=str, default="kpa")
args = parser.parse_args()

autoscaler_type = args.autoscale

if autoscaler_type == "kpa":
    config_path = "./configs/kpa.yml"
elif autoscaler_type == "hpa":
    config_path = "./configs/hpa.yml"
elif autoscaler_type == "apa":
    config_path = "./configs/apa.yml"
    
class InstanceManager:
    def __init__(self, model_name, gpu_name, gpu_number):
        self.model_name = model_name
        self.gpu_name = gpu_name
        self.gpu_number = gpu_number
        self.concurrency_sec_series = None
        
    def monitor_concurrency(self, concurrency_series:TimeSeriesFunction):
        start_timestamp = 0
        end_timestamp = (concurrency_series.timestamps)[-1]
        timestamps = np.arange(start_timestamp, end_timestamp, 1)
        concurrency = np.zeros(len(timestamps))
        
        if concurrency_series.timestamps[0] != 0:
            new_series = np.concatenate([0, concurrency_series.timestamps])
            new_values = np.concatenate([0, concurrency_series.values])
            concurrency_series = TimeSeriesFunction(timestamps=new_series, values=new_values)
        
        i,j = 0,0
        while i in range(len(timestamps)) and j <= len(concurrency_series.timestamps) - 1:
            if timestamps[i] >= (concurrency_series.timestamps)[j + 1]:
                j += 1
                concurrency[i] = (concurrency_series.values)[j]
                i += 1
            else:
                concurrency[i] = (concurrency_series.values)[j]
                i += 1

        self.concurrency_sec_series = TimeSeriesFunction(timestamps=timestamps, values=concurrency)
        return self.concurrency_sec_series
    
    def allocate_instance(self):
        pass
    
    def delete_instance(self):
        pass
        
def monitor_concurrency(concurrency_series:TimeSeriesFunction) -> TimeSeriesFunction:
        start_timestamp = 0
        end_timestamp = (concurrency_series.timestamps)[-1]
        timestamps = np.arange(start_timestamp, end_timestamp + 1, 1)
        concurrency = np.zeros(len(timestamps))
        
        if concurrency_series.timestamps[0] != 0:
            new_series = np.concatenate([[0], concurrency_series.timestamps])
            new_values = np.concatenate([[0], concurrency_series.values])
            concurrency_series = TimeSeriesFunction(timestamps=new_series, values=new_values)
        
        i,j = 0,0
        while i in range(len(timestamps)) and j < len(concurrency_series.timestamps):
            if timestamps[i] >= (concurrency_series.timestamps)[j + 1]:
                j += 1
                concurrency[i] = (concurrency_series.values)[j]
                i += 1
                
            else:
                concurrency[i] = (concurrency_series.values)[j]
                i += 1

        concurrency_sec_series = TimeSeriesFunction(timestamps=timestamps, values=concurrency)
        return concurrency_sec_series
    
def monitor_kvcache_hit_rate(kvcache_hit_rate_series:TimeSeriesFunction) -> TimeSeriesFunction:
        start_timestamp = 0
        end_timestamp = (kvcache_hit_rate_series.timestamps)[-1]
        timestamps = np.arange(start_timestamp, end_timestamp + 1, 1)
        kvcache_hit_rate = np.zeros(len(timestamps))
        
        if kvcache_hit_rate_series.timestamps[0] != 0:
            new_series = np.concatenate([[0], kvcache_hit_rate_series.timestamps])
            new_values = np.concatenate([[0], kvcache_hit_rate_series.values])
            kvcache_hit_rate_series = TimeSeriesFunction(timestamps=new_series, values=new_values)
        
        i,j = 0,0
        while i in range(len(timestamps)) and j < len(kvcache_hit_rate_series.timestamps):
            if timestamps[i] >= (kvcache_hit_rate_series.timestamps)[j + 1]:
                j += 1
                kvcache_hit_rate[i] = (kvcache_hit_rate_series.values)[j]
                i += 1
                
            else:
                kvcache_hit_rate[i] = (kvcache_hit_rate_series.values)[j]
                i += 1

        kvcache_hit_rate_sec_series = TimeSeriesFunction(timestamps=timestamps, values=kvcache_hit_rate)
        return kvcache_hit_rate_sec_series

def concurrency_window(concurrency_series:TimeSeriesFunction, end_timestamp: int, window_size: int=30) -> int:
    start_timestamp = end_timestamp - window_size * 1000 # is ms
    mask = (concurrency_series.timestamps >= start_timestamp) & (concurrency_series.timestamps <= end_timestamp)
    timestamps = concurrency_series.timestamps[mask]
    values = concurrency_series.values[mask]
    window_concurrency_series = TimeSeriesFunction(timestamps=timestamps, values=values)
    concurrency = int(np.ceil(np.mean(window_concurrency_series.values)))
    return concurrency

def get_concurrencys(concurrency_series:TimeSeriesFunction, concurrency_sec_series:TimeSeriesFunction, window_size: int=30, max_workers: int=8) -> np.ndarray[int]:
    timestamps = concurrency_series.timestamps
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        concurrencys = list(executor.map(lambda i: compute(i, concurrency_sec_series, timestamps, window_size), range(len(timestamps))))
    return np.array(concurrencys)

def compute(i, concurrency_sec_series:TimeSeriesFunction, timestamps: np.ndarray, window_size: int) -> int:
    concur = concurrency_window(concurrency_sec_series, timestamps[i], window_size)
    return concur

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def kpa_autoscaler(config, window_size: int = 30):
    current_pod = 1
         
    target_value = config["spec"]["metricsSources"][0]["targetValue"]
    current_pod = config["spec"]["minReplicas"]
    return current_pod

def hpa_autoscaler(config, window_size: int = 30):
    current_pod = 1
         
    target_value = config["spec"]["metricsSources"][0]["targetValue"]
    current_pod = config["spec"]["minReplicas"]
    return current_pod


def apa_autoscaler(config, window_size: int = 30):
    current_pod = 1
         
    target_value = config["spec"]["metricsSources"][0]["targetValue"]
    current_pod = config["spec"]["minReplicas"]
    return current_pod

def save_timeseries_to_json(series: TimeSeriesFunction, output_path: str):
    """
    Save a TimeSeriesFunction to a JSON file.
    Args:
        series: TimeSeriesFunction object containing timestamps and values
        output_path: Path where the JSON file will be saved
    """
    data = {
        "values": series.values.tolist()
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

