import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt


class EventTimestamp:
    def __init__(self, ts, event):
        self.ts = ts
        self.event = event

class TimeSeriesFunction:
    def __init__(self, timestamps, values):
        """
        keypoints: List of (timestamp, value) tuples.
        """
        self.timestamps = np.asarray(timestamps)    
        self.values = np.asarray(values)
        self.interpolator = scipy.interpolate.interp1d(self.timestamps, self.values, kind='linear', fill_value="extrapolate")

    def evaluate(self, t):
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

        ax.plot(self.timestamps, self.values, color="red", label="Key Points", zorder=3)

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
