import pandas as pd

import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import json
from cluster_simulator import PROJECT_DIR
from cluster_simulator import simulate
from cluster_simulator.common import TimeSeriesFunction, EventTimestamp
from cluster_simulator.autoscaler import InstanceManager, concurrency_window, monitor_concurrency, get_concurrencys
from tqdm import tqdm
import time
from typing import Tuple, List

SLO = 2
TRACE_TIME_SPAN = 3600 * 1000 # trace is one hour 3600 * 1000 milliseconds
TIME_DELTA = 0.001 # the value changes in a very short time to simulate pulse function
        
def convert_request_to_event_ts(request_json, model_name, gpu_name) -> Tuple[EventTimestamp, EventTimestamp]:
    arrive_ts = request_json["Request"]["timestamp"] # is ms
    request_data = request_json["Request"]["data"]
    prompt = request_data["prompt"]
    response_length = request_data["max_tokens"]
    prompt_length = len(prompt.split(" "))
    latencys, _,_,_ = simulate(
        model_name=model_name,
        cuda_device_name=gpu_name,
        prompt_length=prompt_length,
        response_length=response_length,
    )
    e2e_latency = sum(latencys) * 1000
    finish_ts = arrive_ts + e2e_latency * SLO

    arrive_event_ts = EventTimestamp(arrive_ts, event="arrive")
    finish_event_ts = EventTimestamp(finish_ts, event="finish")
    
     
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
        
def convert_concurrency_to_chunk_number_series(concurrency_series:TimeSeriesFunction, autoscale: str, target: int) -> TimeSeriesFunction:
    timestamps = concurrency_series.timestamps
    if autoscale == "default":
        print(f"I used default now!!!")
        concurrencys = concurrency_series.values
    elif autoscale == "kpa":
        print(f"I used kpa now!!!")
        concurrency_sec_series = monitor_concurrency(concurrency_series)
        concurrencys = get_concurrencys(concurrency_series, concurrency_sec_series)
    elif autoscale == "hpa":
        print(f"I used hpa now!!!")
        concurrency_sec_series = monitor_concurrency(concurrency_series)
        concurrencys = get_concurrencys(concurrency_series, concurrency_sec_series)
    elif autoscale == "apa":
        print(f"I used apa now!!!")
        concurrency_sec_series = monitor_concurrency(concurrency_series)
        concurrencys = get_concurrencys(concurrency_series, concurrency_sec_series)
    # Notice the times of chunk number don't need 1 more chunk
    chunk_number = np.where((concurrencys >= target) & (concurrencys % target == 0), 
                        concurrencys // target, 
                        concurrencys // target + 1)
    chunk_number_series = TimeSeriesFunction(timestamps=timestamps, values=chunk_number)
    return chunk_number_series

def extract_alloc_free_events(chunk_number_series:TimeSeriesFunction, workload_id: int, chunk_size: int) -> List[EventTimestamp]:
    event_ts_list = []
    previous_gpu_number = 0
    for i in range(len(chunk_number_series.timestamps)):
        timestamp = chunk_number_series.timestamps[i]
        chunk_number = chunk_number_series.values[i]

        if chunk_number == previous_gpu_number:
            continue
        else:
            delta_chunk_number = chunk_number - previous_gpu_number
            event = {
                "workload_id": workload_id,
                "delta_chunk_number": delta_chunk_number,
                "chunk_size": chunk_size,
            }
            event_ts_list.append(EventTimestamp(ts=timestamp, event=event))
        previous_gpu_number = chunk_number
    return event_ts_list

