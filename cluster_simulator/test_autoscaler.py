import os
import unittest
import numpy as np
from common import TimeSeriesFunction
from autoscaler import monitor_concurrency, concurrency_window, get_concurrencys


class TestAutoscaler(unittest.TestCase):
    def test_monitor_concurrency(self):
        # Create a sample TimeSeriesFunction with non-zero start
        timestamps = np.array([0, 2000, 4000, 5000])
        concurrency = np.array([1, 2, 8, 3])
        input_series = TimeSeriesFunction(timestamps=timestamps, values=concurrency)
        
        # Call monitor_concurrency
        result = monitor_concurrency(input_series)
        
        # Verify the output
        self.assertEqual(len(result.timestamps), 5001)
        self.assertEqual(result.timestamps[0], 0)
        self.assertEqual(result.timestamps[-1], 5000)
        
        # Check interpolated values
        self.assertEqual(result.values[0], 1)
        self.assertEqual(result.values[1000], 1)
        self.assertEqual(result.values[2000], 2)
        self.assertEqual(result.values[3000], 2)
        self.assertEqual(result.values[4000], 8)
        self.assertEqual(result.values[5000], 3)
        
    def test_concurrency_window(self):
        timestamps = np.array([0, 10000, 20000, 30000, 40000])
        concurrency = np.array([1, 2, 3, 4, 5])
        input_series = TimeSeriesFunction(timestamps=timestamps, values=concurrency)
        
        result1 = concurrency_window(input_series, end_timestamp=40000, window_size=30)
        self.assertEqual(result1, 4)  # (2+3+4+5)/4 = 3.5
        self.assertEqual(type(result1), int)
        
        result2 = concurrency_window(input_series, end_timestamp=20000, window_size=10)
        self.assertEqual(result2, 3)  # (2+3)/2 = 2.5
        self.assertEqual(type(result2), int)


    def test_get_concurrencys(self):
        timestamps = np.array([0, 20000, 40000, 50000, 100000])
        concurrency = np.array([1, 2, 3, 4, 5])
        concurrency_series = TimeSeriesFunction(timestamps=timestamps, values=concurrency)
        
        sec_timestamps = np.arange(0, 110000, 10000)
        sec_values = np.array([1, 1, 2, 2, 3, 4, 4, 4, 4, 4, 5])
        concurrency_sec_series = TimeSeriesFunction(timestamps=sec_timestamps, values=sec_values)
        
        results1 = get_concurrencys(concurrency_series, concurrency_sec_series, window_size=30)
        
        self.assertEqual(len(results1), len(concurrency), 
                        "Results length should match input timestamps length")
        self.assertEqual(results1[0], 1.0,
                             msg="Average concurrency at t=0 should be 1.0")
        self.assertEqual(results1[1], 2,
                             msg="Average concurrency at t=20000 should be close to 1.33")
        self.assertEqual(results1[2], 2,
                             msg="Average concurrency at t=40000 should be close to 2")
        self.assertEqual(results1[3], 3,
                             msg="Average concurrency at t=50000 should be close to 2")
        self.assertEqual(results1[4], 5,
                             msg="Average concurrency at t=100000 should be close to 5")
        self.assertTrue(all(r >= 0 for r in results1),
                       "All concurrency values should be non-negative")
        
        results2 = get_concurrencys(concurrency_series, concurrency_sec_series, window_size=10)
        self.assertEqual(len(results2), len(concurrency), 
                        "Results length should match input timestamps length")
        self.assertEqual(results2[3], 4,
                             msg="Average concurrency at t=50000 should be close to 3.5")
        self.assertTrue(all(r >= 0 for r in results2),
                       "All concurrency values should be non-negative")


if __name__ == '__main__':
    unittest.main()