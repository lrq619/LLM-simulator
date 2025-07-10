import pandas as pd

import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import json
from cluster_simulator import PROJECT_DIR
from cluster_simulator import simulate
from cluster_simulator.common import TimeSeriesFunction, EventTimestamp
from cluster_simulator.autoscaler import monitor_concurrency, get_concurrencys
from tqdm import tqdm
import time
from typing import Tuple, List, Union

SLO = 2
TRACE_TIME_SPAN = 3600 * 1000 # trace is one hour 3600 * 1000 milliseconds
TIME_DELTA = 0.001 # the value changes in a very short time to simulate pulse function
        
def convert_request_to_event_ts(request_json, model_name, gpu_name) -> Tuple[EventTimestamp, EventTimestamp]:
    arrive_ts = request_json["Request"]["timestamp"] # is ms
    arrive_ts = arrive_ts / 1000
    request_data = request_json["Request"]["data"]
    prompt = request_data["prompt"]
    response_length = request_data["max_tokens"]
    prompt_length = len(prompt.split(" "))
    latencys, _,_,_, request_cache_utilization_list = simulate(
        model_name=model_name,
        cuda_device_name=gpu_name,
        prompt_length=prompt_length,
        response_length=response_length,
    )
    e2e_latency = sum(latencys)
    finish_ts = arrive_ts + e2e_latency * SLO
    arrive_event_ts = EventTimestamp(arrive_ts, event="arrive", request_cache_utilization=request_cache_utilization_list[0])
    finish_event_ts = EventTimestamp(finish_ts, event="finish", request_cache_utilization=request_cache_utilization_list[0])
    
    # utilize_arrive_ts = EventTimestamp(arrive_ts, event="utilize", request_cache_utilization=request_cache_utilization_list[0])
    # utilize_finish_ts = EventTimestamp(finish_ts, event="utilize", request_cache_utilization=request_cache_utilization_list[0])
    
    return arrive_event_ts, finish_event_ts

def convert_processed_trace_to_concurrency_series(processed_trace_file_path, model_name, gpu_name) -> TimeSeriesFunction:
    with open(processed_trace_file_path, 'r') as f:
        json_data = json.load(f)

    event_ts_list = []
    for request_json in tqdm(json_data, desc="Converted", unit="req"):
        arrive_ts, finish_ts = convert_request_to_event_ts(request_json, model_name, gpu_name)
        event_ts_list.append(arrive_ts)
        event_ts_list.append(finish_ts)
    event_ts_list = sorted(event_ts_list,key=lambda event: event.ts)
    # calculate the concurrency
    timestamps = []
    concurrencys = []
    concurrency = 0
    for event_ts in event_ts_list:
        timestamps.append(event_ts.ts)
        if event_ts.event == "arrive":
            concurrency += 1
        elif event_ts.event == "finish":
            concurrency -= 1
        concurrencys.append(concurrency)

    concurrency_series = TimeSeriesFunction(timestamps=timestamps, values=concurrencys)
    return concurrency_series

def convert_processed_trace_to_utilization_series(processed_trace_file_path, model_name, gpu_name) -> TimeSeriesFunction:
    with open(processed_trace_file_path, 'r') as f:
        json_data = json.load(f)
        
    event_ts_list = []
    for request_json in tqdm(json_data, desc="Converted", unit="req"):
            arrive_ts, finish_ts = convert_request_to_event_ts(request_json, model_name, gpu_name)
            event_ts_list.append(arrive_ts)
            event_ts_list.append(finish_ts)
    event_ts_list = sorted(event_ts_list,key=lambda event: event.ts)
    # calculate the utilization
    timestamps = []
    utilizations = []
    utilization = 0
    for event_ts in event_ts_list:
        timestamps.append(event_ts.ts)
        if event_ts.event == "arrive":
            utilization += event_ts.request_cache_utilization
        elif event_ts.event == "finish":
            utilization -= event_ts.request_cache_utilization
        utilizations.append(utilization)
    utilization_series = TimeSeriesFunction(timestamps=timestamps, values=utilizations)
    return utilization_series
      
def convert_concurrency_to_chunk_number_series(concurrency_series:TimeSeriesFunction, autoscale: str, target: Union[int, float]) -> TimeSeriesFunction:
    timestamps = concurrency_series.timestamps
    if autoscale == "default":
        print(f"I used default now!!!")
        concurrencys = concurrency_series.values
    elif autoscale == "kpa" or autoscale == "hpa" or autoscale == "apa":
        print(f"I used {autoscale} now!!!")
        concurrency_sec_series = monitor_concurrency(concurrency_series)
        concurrencys = get_concurrencys(concurrency_series, concurrency_sec_series)
    # Notice the times of chunk number don't need 1 more chunk
    chunk_number = np.where((concurrencys >= target) & (concurrencys % target == 0), 
                        concurrencys // target, 
                        concurrencys // target + 1)
    chunk_number_series = TimeSeriesFunction(timestamps=timestamps, values=chunk_number)
    return chunk_number_series

def extract_alloc_free_events(chunk_number_series:TimeSeriesFunction, workload_id: int, chunk_size: int, node_number: int, gpu_number: int) -> List[EventTimestamp]:
    event_ts_list = []
    previous_gpu_number = 0
    total_gpu_number = node_number * gpu_number
    for i in range(len(chunk_number_series.timestamps)):
        timestamp = chunk_number_series.timestamps[i]
        chunk_number = chunk_number_series.values[i]

        if chunk_number > total_gpu_number:
            chunk_number = total_gpu_number
        
        delta_chunk_number = int(chunk_number - previous_gpu_number)
        if delta_chunk_number == 0:
            continue
        else:
            event = {
                "workload_id": workload_id,
                "delta_chunk_number": delta_chunk_number,
                "chunk_size": chunk_size,
            }
            event_ts_list.append(EventTimestamp(ts=timestamp, event=event))
        previous_gpu_number = chunk_number
    return event_ts_list