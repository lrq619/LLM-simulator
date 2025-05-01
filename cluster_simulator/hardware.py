import numpy as np
from typing import List, Set, Optional, Dict, Tuple
from cluster_simulator.time_series import EventTimestamp, TimeSeriesFunction
class GPUInfo:
    def __init__(self, node_id: int, gpu_id: int):
        self.node_id = node_id
        self.gpu_id = gpu_id

class NodeInfo:
    def __init__(self, node_id: int, gpu_number:int):
        self.orig_gpu_number = gpu_number
        self.node_id = node_id
        self.gpu_set: Set[GPUInfo] = set()
        for i in range(gpu_number):
            gpu = GPUInfo(self.node_id, i)
            self.gpu_set.add(gpu)
    
    def alloc(self, chunk_size: int) -> List[GPUInfo]:
        if self.get_idle_gpu_number() < chunk_size:
            return []
        else:
            gpus = []
            for _ in range(chunk_size):
                gpus.append(self.gpu_set.pop())
            return gpus
    
    def free(self, gpu_info: GPUInfo):
        assert gpu_info.node_id == self.node_id
        self.gpu_set.add(gpu_info)

    def get_idle_gpu_number(self) -> int:
        return len(self.gpu_set)

    def get_cont_gpu_number(self, chunk_size: int) -> int:
        idle_gpu_number = self.get_idle_gpu_number()
        cont_gpu_number = (idle_gpu_number // chunk_size) * chunk_size
        return cont_gpu_number

        

class ClusterManager:
    def __init__(self, node_number: int, gpu_number: int, workload_number: int):
        self.node_dict: Dict[int,NodeInfo] = {}
        for i in range(node_number):
            node = NodeInfo(i, gpu_number)
            self.node_dict[i] = node
        self.workload_gpu_chunks_dict: Dict[int, List[List[GPUInfo]]]={}
        for i in range(workload_number):
            self.workload_gpu_chunks_dict[i] = []

    def alloc(self, chunk_number: int, chunk_size: int) -> List[List[GPUInfo]]:
        # Iterate over nodes and find the first available GPU
        gpu_info_list = []
        for i in range(chunk_number):
            for node_id, node_info in self.node_dict.items():
                gpus = node_info.alloc(chunk_size)
                if gpus == []:
                    continue
                assert len(gpus) == chunk_size
                gpu_info_list.append(gpus)
                break
        return gpu_info_list


    def free(self, gpu_chunks: List[List[GPUInfo]]):
        for gpu_chunk in gpu_chunks:
            for gpu_info in gpu_chunk:
                node_info = self.node_dict[gpu_info.node_id]
                node_info.free(gpu_info)

    def get_idle_gpu_number(self) -> int:
        idle_gpu_number = 0
        for node_id, node_info in self.node_dict.items():
            idle_gpu_number += node_info.get_idle_gpu_number()
        return idle_gpu_number

    def get_cont_gpu_number(self, chunk_size: int) -> int:
        cont_gpu_number = 0
        for node_id, node_info in self.node_dict.items():
            cont_gpu_number += node_info.get_cont_gpu_number(chunk_size)
        return cont_gpu_number
            

    def replay(self, gpu_operations: List[EventTimestamp], max_chunk_size: int) -> Tuple[TimeSeriesFunction, TimeSeriesFunction, List[EventTimestamp]]:
        # sort the operations based on timestamp
        gpu_operations = sorted(gpu_operations, key=lambda event: event.ts)
        idle_gpu_number = 0
        idle_gpu_numbers = []

        cont_gpu_number = 0
        cont_gpu_numbers = []
        timestamps = []
        alloc_events : List[EventTimestamp] = []
        # replay the gpu operations
        for gpu_operation in gpu_operations:
            event = gpu_operation.event
            workload_id = event["workload_id"]
            delta_chunk_number = event["delta_chunk_number"]
            chunk_size = event["chunk_size"]
            assert delta_chunk_number != 0
            if delta_chunk_number > 0:
                gpu_chunks = self.alloc(delta_chunk_number, chunk_size)
                for gpu_chunk in gpu_chunks:
                    self.workload_gpu_chunks_dict[workload_id].append(gpu_chunk)
                # identify whether this allocation succeeds or not
                if len(gpu_chunks) < delta_chunk_number:
                    event = {
                        "workload_id": workload_id,
                        "chunk_size": chunk_size,
                        "delta_chunk_number": delta_chunk_number,
                        "success": False,
                        "idle_gpu_number": self.get_idle_gpu_number(),
                    }
                else:
                    event = {
                        "workload_id": workload_id,
                        "chunk_size": chunk_size,
                        "delta_chunk_number": delta_chunk_number,
                        "success": True,
                        "idle_gpu_number": self.get_idle_gpu_number(),
                    }
                alloc_events.append(EventTimestamp(ts=gpu_operation.ts, event=event))

            else:
                gpu_chunks = []
                workload_gpu_chunk_set = self.workload_gpu_chunks_dict[workload_id]
                for i in range(-delta_chunk_number):
                    if len(workload_gpu_chunk_set) == 0:
                        break
                    gpu_chunks.append(workload_gpu_chunk_set.pop())
                self.free(gpu_chunks)

            idle_gpu_number = self.get_idle_gpu_number()
            cont_gpu_number = self.get_cont_gpu_number(chunk_size=max_chunk_size)
            idle_gpu_numbers.append(idle_gpu_number)
            cont_gpu_numbers.append(cont_gpu_number)
            timestamps.append(gpu_operation.ts)


        idle_gpu_number_series = TimeSeriesFunction(timestamps, idle_gpu_numbers)
        cont_gpu_number_series = TimeSeriesFunction(timestamps, cont_gpu_numbers)
        return idle_gpu_number_series, cont_gpu_number_series, alloc_events
            

