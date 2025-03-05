from typing import List, Set, Optional, Dict, Tuple
from cluster_simulator.time_series import EventTimestamp, TimeSeriesFunction
class GPUInfo:
    def __init__(self, node_id: int, gpu_id: int):
        self.node_id = node_id
        self.gpu_id = gpu_id

class NodeInfo:
    def __init__(self, node_id: int, gpu_number:int):
        self.node_id = node_id
        self.gpu_set: Set[GPUInfo] = set()
        for i in range(gpu_number):
            gpu = GPUInfo(self.node_id, i)
            self.gpu_set.add(gpu)
    
    def alloc(self) -> Optional[GPUInfo]:
        if len(self.gpu_set) == 0:
            return None
        else:
            return self.gpu_set.pop()
    
    def free(self, gpu_info: GPUInfo):
        assert gpu_info.node_id == self.node_id
        self.gpu_set.add(gpu_info)

    def get_idle_gpu_number(self) -> int:
        return len(self.gpu_set)

class ClusterManager:
    def __init__(self, node_number: int, gpu_number: int, workload_number: int):
        self.node_dict: Dict[int,NodeInfo] = {}
        for i in range(node_number):
            node = NodeInfo(i, gpu_number)
            self.node_dict[i] = node
        self.workload_gpu_set_dict: Dict[int, Set[GPUInfo]]={}
        for i in range(workload_number):
            self.workload_gpu_set_dict[i] = set()

    def alloc(self, gpu_number: int) -> List[GPUInfo]:
        # Iterate over nodes and find the first available GPU
        gpu_info_list = []
        for i in range(gpu_number):
            for node_id, node_info in self.node_dict.items():
                gpu_info = node_info.alloc()
                if gpu_info == None:
                    continue
                gpu_info_list.append(gpu_info)
                break
        return gpu_info_list


    def free(self, gpu_infos: List[GPUInfo]):
        for gpu_info in gpu_infos:
            node_info = self.node_dict[gpu_info.node_id]
            node_info.free(gpu_info)

    def get_idle_gpu_number(self) -> int:
        idle_gpu_number = 0
        for node_id, node_info in self.node_dict.items():
            idle_gpu_number += node_info.get_idle_gpu_number()
        return idle_gpu_number

    def get_max_tp_level(self) -> int:
        max_tp_level = 0
        for node_id, node_info in self.node_dict.items():
            max_tp_level = max(max_tp_level, node_info.get_idle_gpu_number())
        return max_tp_level

    def replay(self, gpu_operations: List[EventTimestamp]) -> Tuple[TimeSeriesFunction, TimeSeriesFunction]:
        # sort the operations based on timestamp
        gpu_operations = sorted(gpu_operations, key=lambda event: event.ts)
        idle_gpu_number = 0
        idle_gpu_numbers = []

        max_tp_level = 0
        max_tp_levels = []
        timestamps = []
        # replay the gpu operations
        for gpu_operation in gpu_operations:
            event = gpu_operation.event
            splits = event.split(':')
            assert len(splits) == 2
            workload_id = int(splits[0])
            delta_gpu_number = int(splits[1])
            assert delta_gpu_number != 0
            if delta_gpu_number > 0:
                gpu_info_list = self.alloc(delta_gpu_number)
                for gpu_info in gpu_info_list:
                    self.workload_gpu_set_dict[workload_id].add(gpu_info)
            else:
                gpu_info_list = []
                workload_gpu_set = self.workload_gpu_set_dict[workload_id]
                for i in range(-delta_gpu_number):
                    if len(workload_gpu_set) == 0:
                        break
                    gpu_info_list.append(workload_gpu_set.pop())
                self.free(gpu_info_list)

                idle_gpu_number = self.get_idle_gpu_number()
                max_tp_level = self.get_max_tp_level()
                idle_gpu_numbers.append(idle_gpu_number)
                max_tp_levels.append(max_tp_level)
                timestamps.append(gpu_operation.ts)


        idle_gpu_number_series = TimeSeriesFunction(timestamps, idle_gpu_numbers)
        max_tp_level_series = TimeSeriesFunction(timestamps, max_tp_levels)
        return idle_gpu_number_series, max_tp_level_series
            

