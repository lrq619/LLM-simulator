import pytest
from cluster_simulator.hardware import GPUInfo, NodeInfo, ClusterManager
from cluster_simulator.time_series import EventTimestamp

def test_gpu_info():
    gpu = GPUInfo(node_id=1, gpu_id=2)
    assert gpu.node_id == 1
    assert gpu.gpu_id == 2

class TestNodeInfo:
    def test_init(self):
        node0 = NodeInfo(node_id=0, gpu_number=4)
        node1 = NodeInfo(node_id=1, gpu_number=4)
        assert node0.node_id == 0
        assert node0.orig_gpu_number == 4
        assert len(node0.gpu_set) == 4
        
        # Verify all GPUs have correct node_id and unique gpu_ids
        gpu_ids = set()
        for gpu in node0.gpu_set:
            assert gpu.node_id == 0
            gpu_ids.add(gpu.gpu_id)
        assert len(gpu_ids) == 4

        assert node1.node_id == 1
        assert node1.orig_gpu_number == 4
        assert len(node1.gpu_set) == 4
        
    def test_alloc(self):
        node = NodeInfo(node_id=0, gpu_number=4)
        
        # Test successful allocation
        gpus = node.alloc(chunk_size=2)
        assert len(gpus) == 2
        assert len(node.gpu_set) == 2
        
        # Test allocation when not enough GPUs
        gpus = node.alloc(chunk_size=3)
        assert len(gpus) == 0
        assert len(node.gpu_set) == 2

    def test_free(self):
        node = NodeInfo(node_id=0, gpu_number=4)
        gpus = node.alloc(chunk_size=2)
        initial_gpu_count = len(node.gpu_set)
        
        node.free(gpus[0])
        assert len(node.gpu_set) == initial_gpu_count + 1

    def test_get_idle_gpu_number(self):
        node = NodeInfo(node_id=0, gpu_number=4)
        assert node.get_idle_gpu_number() == 4
        
        node.alloc(chunk_size=2)
        assert node.get_idle_gpu_number() == 2

    def test_get_cont_gpu_number(self):
        node = NodeInfo(node_id=0, gpu_number=4)
        assert node.get_cont_gpu_number(chunk_size=2) == 4
        assert node.get_cont_gpu_number(chunk_size=3) == 3
        
        node.alloc(chunk_size=1)
        assert node.get_cont_gpu_number(chunk_size=2) == 2

class TestClusterManager:
    @pytest.fixture
    def cluster(self):
        return ClusterManager(node_number=2, gpu_number=4, workload_number=3)

    def test_init(self, cluster):
        assert len(cluster.node_dict) == 2
        assert len(cluster.workload_gpu_chunks_dict) == 3
        for node in cluster.node_dict.values():
            assert node.get_idle_gpu_number() == 4

    def test_alloc(self, cluster):
        # Test successful allocation
        gpu_chunks = cluster.alloc(chunk_number=2, chunk_size=2)
        assert len(gpu_chunks) == 2
        assert all(len(chunk) == 2 for chunk in gpu_chunks)
        
        # Test allocation when resources are exhausted
        gpu_chunks = cluster.alloc(chunk_number=5, chunk_size=2)
        assert len(gpu_chunks) < 5

    def test_free(self, cluster):
        gpu_chunks = cluster.alloc(chunk_number=2, chunk_size=2)
        initial_idle = cluster.get_idle_gpu_number()
        
        cluster.free(gpu_chunks)
        assert cluster.get_idle_gpu_number() == initial_idle + 4

    def test_get_idle_gpu_number(self, cluster):
        assert cluster.get_idle_gpu_number() == 8
        
        cluster.alloc(chunk_number=1, chunk_size=2)
        assert cluster.get_idle_gpu_number() == 6

    def test_get_cont_gpu_number(self, cluster):
        assert cluster.get_cont_gpu_number(chunk_size=2) == 8
        assert cluster.get_cont_gpu_number(chunk_size=3) == 6

    def test_replay(self, cluster):
        # Create sample GPU operations
        operations = [
            EventTimestamp(ts=0, event={
                "workload_id": 0,
                "delta_chunk_number": 2,
                "chunk_size": 2
            }),
            EventTimestamp(ts=1, event={
                "workload_id": 0,
                "delta_chunk_number": -1,
                "chunk_size": 2
            })
        ]

        idle_series, cont_series, alloc_events = cluster.replay(
            gpu_operations=operations,
        )

        # Test returned time series
        assert len(idle_series.timestamps) == 2
        assert len(cont_series.timestamps) == 2
        assert len(alloc_events) == 1  # Only allocation events are recorded

        # Verify allocation event
        assert alloc_events[0].event["success"] == True
        assert alloc_events[0].event["workload_id"] == 0
        assert alloc_events[0].event["delta_chunk_number"] == 2

def test_integration():
    # Create a cluster
    cluster = ClusterManager(node_number=2, gpu_number=4, workload_number=2)
    
    # Test allocation and freeing
    gpu_chunks = cluster.alloc(chunk_number=3, chunk_size=2)
    assert cluster.get_idle_gpu_number() == 2  # 8 total - 6 allocated
    
    cluster.free(gpu_chunks)
    assert cluster.get_idle_gpu_number() == 8  # All GPUs free again
    
    # Test replay functionality
    operations = [
        EventTimestamp(ts=0, event={"workload_id": 0, "delta_chunk_number": 2, "chunk_size": 2}),
        EventTimestamp(ts=1, event={"workload_id": 1, "delta_chunk_number": 1, "chunk_size": 2}),
        EventTimestamp(ts=2, event={"workload_id": 0, "delta_chunk_number": -2, "chunk_size": 2})
    ]
    
    idle_series, cont_series, alloc_events = cluster.replay(operations)
    assert len(idle_series.timestamps) == 3
    assert len(alloc_events) == 2  # Two allocation events