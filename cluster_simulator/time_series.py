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
from typing import Tuple, List, Union
from transformers import AutoTokenizer

SLO = 2
TRACE_TIME_SPAN = 3600 * 1000 # trace is one hour 3600 * 1000 milliseconds
TIME_DELTA = 0.001 # the value changes in a very short time to simulate pulse function
        
def convert_request_to_event_ts(request_json, model_name, gpu_name) -> Tuple[EventTimestamp, EventTimestamp]:
    arrive_ts = request_json["Request"]["timestamp"] * TIME_DELTA # is ms
    request_data = request_json["Request"]["data"]
    prompt = request_data["prompt"]
    # # TODO: increase the efficiency
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # prompt_length = len(tokenizer.encode(prompt))
    prompt_length = len(prompt.split(" "))
    response_length = request_data["min_tokens"]
    
    latencys, _,_,_,_,_ = simulate(
        model_name=model_name,
        cuda_device_name=gpu_name,
        prompt_length=prompt_length,
        response_length=response_length,
    )
    e2e_latency = sum(latencys)
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

def convert_processed_trace_to_memory(request_json, model_name, gpu_name, kvc_size_GB) -> dict:
    arrive_ts = request_json["Request"]["timestamp"] * TIME_DELTA # here is ms
    request_data = request_json["Request"]["data"]
    prompt = request_data["prompt"]
    # # TODO: increase the efficiency
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # prompt_length = len(tokenizer.encode(prompt))
    prompt_length = len(prompt.split(" ")) 
    response_length = request_data["min_tokens"]
    
    latencys, _,_,_,_,_  = simulate(
        model_name=model_name,
        cuda_device_name=gpu_name,
        prompt_length=prompt_length,
        response_length=response_length,
    )
    ttft = latencys[0]
    e2e_latency = sum(latencys) * SLO

    finish_ts = arrive_ts + e2e_latency 
    base_kvc_memory_gb = prompt_length * kvc_size_GB
    decode_kvc_memory_increase_gb = response_length * kvc_size_GB
    return {
        'request_id': request_json["GlobalID"],
        'start': arrive_ts,
        'finish': finish_ts,
        'ttft': ttft,
        'prompt_kvc_gb': base_kvc_memory_gb,
        'decode_kvc_increase_gb': decode_kvc_memory_increase_gb,
        'total_duration': e2e_latency
    }

def convert_processed_trace_to_utilization_series(processed_trace_file_path: str, model_name: str,
                                                       gpu_name: str,
                                                       kvc_size_gb_per_token: float
                                                       ) -> TimeSeriesFunction:
    
    with open(processed_trace_file_path, 'r') as f:
        json_data = json.load(f)

    requests_memory_profiles = []
    min_global_time = float('inf')
    max_global_time = 0

    for request_json in tqdm(json_data, desc="Estimating Request Memory Profiles", unit="req"):
        try:
            mem_profile = convert_processed_trace_to_memory(
                request_json, model_name, gpu_name, kvc_size_gb_per_token)
            requests_memory_profiles.append(mem_profile)
            min_global_time = min(min_global_time, mem_profile['start'])
            max_global_time = max(max_global_time, mem_profile['finish'])
        except Exception as e:
            print(f"Error processing request for memory profile: {e}, request_json: {request_json}")
            continue

    if not requests_memory_profiles:
        return TimeSeriesFunction([], [])
    
    event_timestamps_ms = set()
    for req_profile in requests_memory_profiles:
        event_timestamps_ms.add(req_profile['start'])
        if req_profile['ttft'] > 0:
             event_timestamps_ms.add(req_profile['start'] + req_profile['ttft'])
        event_timestamps_ms.add(req_profile['finish'])
        
    sampled_timestamps_ms = np.array(sorted(list(event_timestamps_ms)))
    sampled_total_memory_gb = []
    for current_sample_ts_ms in tqdm(sampled_timestamps_ms, desc="Aggregating Memory Usage", unit="ms"):
        total_memory_at_this_ts_gb = 0.0
        for req_profile in requests_memory_profiles:
            if req_profile['start'] <= current_sample_ts_ms < req_profile['finish']:
                req_current_kvc_memory_gb = 0.0
                if current_sample_ts_ms < req_profile['start'] + req_profile['ttft']:
                    req_current_kvc_memory_gb = req_profile['prompt_kvc_gb']
                else:
                    time_in_decode = current_sample_ts_ms - (req_profile['start'] + req_profile['ttft'])
                    decode_duration = req_profile['total_duration'] - req_profile['ttft']
                    memory_growth_ratio = min(1.0, max(0.0, time_in_decode / decode_duration))
                    req_current_kvc_memory_gb = req_profile['prompt_kvc_gb'] + (req_profile['decode_kvc_increase_gb'] * memory_growth_ratio)
                total_memory_at_this_ts_gb += req_current_kvc_memory_gb
        sampled_total_memory_gb.append(total_memory_at_this_ts_gb)

    memory_series = TimeSeriesFunction(timestamps=sampled_timestamps_ms, values=sampled_total_memory_gb)
    return memory_series

def convert_concurrency_to_chunk_number_series(concurrency_series:TimeSeriesFunction, autoscale: str, target: Union[int, float]) -> TimeSeriesFunction:
    timestamps = concurrency_series.timestamps
    if autoscale in ["default", "hpa"]:
        concurrencys = concurrency_series.values
    elif autoscale == "kpa":
        concurrency_sec_series = monitor_concurrency(concurrency_series)
        concurrencys = get_concurrencys(concurrency_series, concurrency_sec_series)
    elif autoscale == "apa":
        tolerance_factor = 2
        concurrencys = concurrency_series.values
        target = tolerance_factor * target
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