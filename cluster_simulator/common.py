import numpy as np
import scipy
import json
from tqdm import tqdm
from simulate import simulate
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import requests
class EventTimestamp:
    def __init__(self, ts, event, request_cache_utilization: Optional[float]=None):
        self.ts = ts
        self.event = event
        self.request_cache_utilization = request_cache_utilization

class TimeSeriesFunction:
    def __init__(self, timestamps, values):
        """
        keypoints: List of (timestamp, value) tuples.
        """
        self.timestamps = np.asarray(timestamps)    
        self.values = np.asarray(values)
        self.interpolator = scipy.interpolate.interp1d(self.timestamps, self.values, kind='linear', fill_value="extrapolate")
        self.min_timestamp = min(self.timestamps)
        self.max_timestamp = max(self.timestamps)

    def evaluate(self, t):
        if t < self.min_timestamp or t > self.max_timestamp:
            return 0
        return float(self.interpolator(t))

    def sample(self,num_points=100):
        min_timestamp = min(self.timestamps)
        max_timestamp = max(self.timestamps)
        delta_time = (max_timestamp - min_timestamp) / num_points
        sampled_timestamps = [min_timestamp + delta_time * (i+1) for i in range(num_points)]
        sampled_values = [self.evaluate(t) for t in sampled_timestamps]
        return sampled_timestamps, sampled_values


    def plot(self, num_points=100):
        """Plot the interpolated function and return a matplotlib.figure.Figure instance."""
        fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(self.timestamps, self.values, color="red", label="vllm", zorder=3)

        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True)

        return fig  # Return the figure instance

    def __repr__(self):
        return repr(self.data)

    def __add__(self, other):
    # Ensure other is compatible
        if not isinstance(other, TimeSeriesFunction):
            raise TypeError("Can only add TimeSeriesFunction to TimeSeriesFunction")

        merged_timestamps = np.concatenate((self.timestamps, other.timestamps))
        merged_timestamps = np.unique(merged_timestamps)
        merged_timestamps.sort()
        merged_values = []
        for ts in merged_timestamps:
            value_self = self.evaluate(ts)
            value_other = other.evaluate(ts)
            merged_values.append(value_self+value_other)

        return TimeSeriesFunction(merged_timestamps, merged_values)

def fetch_data(params_list, url):
    data = {}
    for params in params_list:
        response = requests.get(url, params=params)
        response_json = response.json()
        print(response_json) 
        for result in response_json.get("data", {}).get("result", []):
            gpu_uuid = result["metric"].get("gpuIDs", "unknown")
            hostname = result['metric'].get("hostname", "unknown")
            instance_uuid = f"{hostname}-{gpu_uuid}"
            # metric_name = result["metric"]["__name__"]
            metric_name: str = params["query"]
            if metric_name.startswith("router"):
                instance_uuid = "router"
            
            if instance_uuid not in data:
                data[instance_uuid] = {}
            
            if metric_name not in data[instance_uuid]:
                data[instance_uuid][metric_name] = [] 
            data[instance_uuid][metric_name] += result["values"]
    return data