import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pytest
import numpy as np
from cluster_simulator.time_series import (
    EventTimestamp,
    TimeSeriesFunction,
    convert_request_to_event_ts,
    convert_processed_trace_to_concurrency_series,
    convert_concurrency_to_chunk_number_series,
    extract_alloc_free_events, 
    monitor_concurrency,
    get_concurrencys,
)

# Test EventTimestamp class
def test_event_timestamp():
    ts = 1000
    event = "test_event"
    event_ts = EventTimestamp(ts, event)
    assert event_ts.ts == ts
    assert event_ts.event == event

# Test TimeSeriesFunction class
def test_time_series_function():
    timestamps = [0, 1, 2]
    values = [10, 20, 30]
    tsf = TimeSeriesFunction(timestamps, values)
    
    # Test evaluate
    assert tsf.evaluate(0) == 10
    assert tsf.evaluate(1) == 20
    assert tsf.evaluate(1.5) == 25  # Linear interpolation
    
    # Test sample
    sampled_ts, sampled_vals = tsf.sample(num_points=5)
    assert len(sampled_ts) == 5
    assert len(sampled_vals) == 5

# Test convert_request_to_event_ts
def test_convert_request_to_event_ts():
    
    request_json = {
        "GlobalID":{
            "global_id": 0
        },
        "Request":{
            "timestamp": 26,
            "data": {
                "prompt": "test prompt",
                "max_response_length": 100,
                "model_name": "meta-llama/Meta-Llama-3-8B",
                "global_request_id": 0
        },
        "attributes": {
            "global_id": 0
        }
            }
        }
    
    gpu_name = "NVIDIA A100-SXM4-40GB"
    model_name = "meta-llama/Meta-Llama-3-8B"
    arrive_ts, finish_ts = convert_request_to_event_ts(request_json, model_name, gpu_name)
    
    assert arrive_ts.ts == 26
    assert arrive_ts.event == "arrive"
    assert finish_ts.ts > arrive_ts.ts
    assert finish_ts.event == "finish"

# Test convert_processed_trace_to_concurrency_series
@pytest.fixture
def sample_trace_file(tmp_path):
    data = [
        {
            "timestamp": 0,
            "data": {
            "prompt": "test prompt 1",
            "max_response_length": 44,
            "model_name": "meta-llama/Meta-Llama-3-8B",
            "global_request_id": 0
            },
            "attributes": {
            "global_id": 0
            }
        },
        {
            "timestamp": 2000,
            "data": {
                "prompt": "test prompt 2",
                "max_response_length": 100,
                "model_name": "meta-llama/Meta-Llama-3-8B",
                "global_request_id": 1
            },
            "attributes": {
            "global_id": 1
            }
        },
        {
            "timestamp": 3000,
            "data": {
                "prompt": "test prompt 2",
                "max_response_length": 140,
                "model_name": "meta-llama/Meta-Llama-3-8B",
                "global_request_id": 2
            },
            "attributes": {
            "global_id": 2
            }
        },
    ]
    file_path = tmp_path / "test_trace.json"
    with open(file_path, 'w') as f:
        json.dump(data, f)
    return str(file_path)

def test_convert_processed_trace_to_concurrency_series(sample_trace_file):
    gpu_name = "NVIDIA A100-SXM4-40GB"
    model_name = "meta-llama/Meta-Llama-3-8B"
    concurrency_series = convert_processed_trace_to_concurrency_series(sample_trace_file, model_name, gpu_name)
    
    assert isinstance(concurrency_series, TimeSeriesFunction)
    assert len(concurrency_series.timestamps) > 0
    assert len(concurrency_series.values) > 0

# Test convert_concurrency_to_chunk_number_series
def test_convert_concurrency_to_chunk_number_series():
    timestamps = [0, 1000, 2000, 3000]
    concurrencies = [1, 3, 5, 2]
    concurrency_series = TimeSeriesFunction(timestamps, concurrencies)
    target = 2
    
    chunk_series = convert_concurrency_to_chunk_number_series(concurrency_series, target)
    
    assert isinstance(chunk_series, TimeSeriesFunction)
    assert all(chunk_series.values == np.array([1, 2, 3, 1]))

# Test extract_alloc_free_events
def test_extract_alloc_free_events():
    timestamps = [0, 1000, 2000, 3000, 4000]
    chunk_numbers = [0, 1, 2, 2, 1]
    chunk_series = TimeSeriesFunction(timestamps, chunk_numbers)
    workload_id = 1
    chunk_size = 4
    node_number = 1
    gpu_number = 4
    
    events = extract_alloc_free_events(chunk_series, workload_id, chunk_size, node_number, gpu_number)
    assert events[0].event["workload_id"] == 1
    assert len(events) == 3  # Should have 3 changes in chunk numbers
    assert events[0].ts == 1000  # First change at t=1
    assert events[1].ts == 2000  # Second change at t=2
    assert events[2].ts == 4000  # Third change at t=2
    assert events[0].event["delta_chunk_number"] == 1
    assert events[1].event["delta_chunk_number"] == 1
    assert events[2].event["delta_chunk_number"] == -1
    assert chunk_numbers[1] - events[0].event["delta_chunk_number"] <= 4
    
def test_monitor_concurrency():
    timestamps = [0, 1000, 2000, 3000, 4000]
    concurrencies = [1, 3, 5, 2, 1]
    concurrency_series = TimeSeriesFunction(timestamps, concurrencies)
    concurrency_sec_series = monitor_concurrency(concurrency_series)
    assert isinstance(concurrency_sec_series, TimeSeriesFunction)
    assert len(concurrency_sec_series.timestamps) > 0
    assert len(concurrency_sec_series.values) > 0
    
def test_get_concurrencys():
    timestamps = [0, 1000, 2000, 3000, 4000]
    concurrencies = [1, 3, 5, 2, 1]
    concurrency_series = TimeSeriesFunction(timestamps, concurrencies)
    concurrency_sec_series = monitor_concurrency(concurrency_series)
    concurrencys = get_concurrencys(concurrency_series, concurrency_sec_series)
    assert isinstance(concurrencys, np.ndarray)
    assert len(concurrencys) > 0
    assert len(concurrencys) == len(concurrency_series.timestamps)