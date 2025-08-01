import pytest
from cluster_simulator.main import process_alloc_events, get_failure_alloc_series
from cluster_simulator.time_series import TimeSeriesFunction, EventTimestamp

class TestMain:
    @pytest.fixture
    def sample_alloc_events(self):
        """Create sample allocation events for testing"""
        events = [
            EventTimestamp(
                ts=1.0,
                event={
                    "workload_id": 0,
                    "success": True,
                    "chunk_size": 2,
                    "delta_chunk_number": 1,
                    "idle_gpu_number": 8
                }
            ),
            EventTimestamp(
                ts=2.0,
                event={
                    "workload_id": 0,
                    "success": False,
                    "chunk_size": 4,
                    "delta_chunk_number": 1,
                    "idle_gpu_number": 6
                }
            ),
            EventTimestamp(
                ts=3.0,
                event={
                    "workload_id": 1,
                    "success": False,
                    "chunk_size": 2,
                    "delta_chunk_number": 1,
                    "idle_gpu_number": 3
                }
            )
        ]
        return events

    def test_process_alloc_events(self, sample_alloc_events, capsys):
        """Test process_alloc_events function"""
        process_alloc_events(sample_alloc_events)
        captured = capsys.readouterr()
        
        # Check if the output contains expected information
        assert "For workload: 0" in captured.out
        assert "For workload: 1" in captured.out
        assert "totally 2 allocs" in captured.out  # For workload 0
        assert "totally 1 allocs" in captured.out  # For workload 1
        assert "1 are failed" in captured.out      # For workload 0
        assert "1 are failed" in captured.out      # For workload 1


    def test_get_failure_alloc_series(self, sample_alloc_events):
        """Test get_failure_alloc_series function"""
        result = get_failure_alloc_series(sample_alloc_events)
        
        # Check if result is TimeSeriesFunction
        assert isinstance(result, TimeSeriesFunction)
        
        # Check timestamps
        assert list(result.timestamps) == [1.0, 2.0, 3.0]
        
        # Check values (0 for success, idle_gpu_number for failure)
        assert list(result.values) == [0, 6, 3]

    def test_get_failure_alloc_series_both_cases(self):
        """Test get_failure_alloc_series with both successful and failed allocations"""
        events = [
            EventTimestamp(
                ts=1.0,
                event={
                    "success": True,
                    "idle_gpu_number": 8
                }
            ),
            EventTimestamp(
                ts=2.0,
                event={
                    "success": True,
                    "idle_gpu_number": 6
                }
            ),
            EventTimestamp(
                ts=3.0,
                event={
                    "success": False,
                    "idle_gpu_number": 4
                }
            )
        ]
        
        result = get_failure_alloc_series(events)
        assert list(result.timestamps) == [1.0, 2.0, 3.0]
        assert list(result.values) == [0, 0, 4]
