import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cluster_simulator.time_series import TimeSeriesFunction
# Add the project root directory to Python path

def test_normalize_ts():
    from cluster_simulator.autoscaler import normalize_ts
    timestamps = np.array([
                0.0,
                1.0,
                1.1,
                1.7522478426493552,
                2.0123,
                2.342,
                2.8657,
                4.0234771749939922,
                4.9747200765090238,
                5.0])
    values = np.array([1, 2, 3, 4, 3 , 2, 1, 0, 0, 2])
    input_series = TimeSeriesFunction(timestamps=timestamps, values=values)
    result = normalize_ts(input_series)
    assert len(result.timestamps) == 3600
    assert result.timestamps[0] == 0
    assert result.timestamps[-1] == 3599
    assert result.values[0] == 1
    assert result.values[-1] == 2
    assert len(result.values) == 3600
test_normalize_ts()

def test_extract_alloc_free_events():
    from cluster_simulator.time_series import extract_alloc_free_events
    timestamps = np.array([0,1,2,3,4])
    chunk_number = np.array([0, 1, 2, 2, 1])
    input_series = TimeSeriesFunction(timestamps=timestamps, values=chunk_number)
    events = extract_alloc_free_events(input_series, 0, 1, 1, 4)
    assert events[0].event["workload_id"] == 0
    assert len(events) == 3  # Should have 3 changes in chunk numbers
    assert events[0].ts == 1  # First change at t=1
    assert events[2].ts == 4  # Third change at t=2
    assert events[0].event["delta_chunk_number"] == 1
    assert events[2].event["delta_chunk_number"] == -1
    assert chunk_number[1] - events[0].event["delta_chunk_number"] <= 4
    
test_extract_alloc_free_events()

