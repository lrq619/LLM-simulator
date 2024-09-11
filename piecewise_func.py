class PiecewiseLinear:
    "Piecewise Linear function for simulating a single request's memory usage. "
    "Take starting timestamp, prompt and total memory occupancy, prompt and total latency as input"
    "Will calculate 3 points, X axis is in second, Y axis is in GB"
    def __init__(self, start_x: float, prompt_mem: float, prompt_latency: float, total_mem: float, total_latency: float):
        start_y = 0.0
        prompt_x = start_x + prompt_latency
        finish_x = start_x + total_latency
        prompt_y = prompt_mem
        finish_y = total_mem
        
        self.x_array = [start_x, prompt_x, finish_x]
        self.y_array = [start_y, prompt_y, finish_y]
        
    def interpolate(self, x: float):
        if x < self.x_array[0] or x > self.x_array[2]:
            return 0
        
        if x == self.x_array[0]:
            return self.y_array[0]
        elif x == self.x_array[1]:
            return self.y_array[1]
        elif x == self.x_array[2]:
            return self.y_array[2]
        
        if x <= self.x_array[1]:
            return self._interpolate(x, self.x_array[0], self.x_array[1], self.y_array[0], self.y_array[1])
        else:
            return self._interpolate(x, self.x_array[1], self.x_array[2], self.y_array[1], self.y_array[2])
            
            
    def _interpolate(self, x, x1, x2, y1, y2):
        return y1 + (y2 - y1)/(x2 - x1)*(x - x1)