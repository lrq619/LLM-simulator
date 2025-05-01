import pytest
import os
import json
from unittest.mock import patch
from simulate import get_gpu_info, get_model_info, get_ptps, simulate
from utils import PROJECT_ROOT_PATH

# Mock data for testing
MOCK_GPU_DATA = {
    "NVIDIA A100-SXM4-40GB": {
        "memory_bw": 1500.0,
        "memory_bw_util": 60.0,
        "memory_cap": 40
    }
}

MOCK_MODEL_DATA = {
    "llama-8b": {
        "num_hidden_layers": 32,
        "num_kv_heads": 8,
        "model_size_GB": 16.0,
        "kvc_size_KB": 128.0
    }
}

MOCK_PTPS_DATA = {
    "llama-8b": {
        "NVIDIA A100-SXM4-40GB": 2500.0
    }
}

@pytest.fixture
def setup_mock_files(tmp_path):
    # Create data directory in tmp_path
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    assert data_dir.exists()
    
    # Create temporary JSON files with mock data
    gpu_file = data_dir / "gpu.json"
    model_file = data_dir / "model.json"
    ptps_file = data_dir / "ptps.json"
    
    
    gpu_file.write_text(json.dumps(MOCK_GPU_DATA))
    model_file.write_text(json.dumps(MOCK_MODEL_DATA))
    ptps_file.write_text(json.dumps(MOCK_PTPS_DATA))
    
    return str(tmp_path)

def test_get_gpu_info(setup_mock_files):
    with patch("simulate.PROJECT_ROOT_PATH", setup_mock_files):
        memory_cap, memory_bw, memory_bw_util = get_gpu_info("NVIDIA A100-SXM4-40GB")
        
        assert memory_cap == 40
        assert memory_bw == 1500.0
        assert memory_bw_util == 60.0

def test_get_model_info(setup_mock_files):
    with patch("simulate.PROJECT_ROOT_PATH", setup_mock_files):
        num_hidden_layers, num_kv_heads, model_size_GB, kvc_size_KB = get_model_info("llama-8b")
        
        assert num_hidden_layers == 32
        assert num_kv_heads == 8
        assert model_size_GB == 16.0
        assert kvc_size_KB == 128.0

def test_get_ptps(setup_mock_files):
    with patch("simulate.PROJECT_ROOT_PATH", setup_mock_files):
        ptps = get_ptps("llama-8b", "NVIDIA A100-SXM4-40GB")
        
        assert ptps == 2500.0

def test_simulate(setup_mock_files):
    with patch("simulate.PROJECT_ROOT_PATH", setup_mock_files):
        # Test simulate function
        latencies, alpha, beta, c = simulate(
            model_name="llama-8b",
            cuda_device_name="NVIDIA A100-SXM4-40GB",
            prompt_length=1024,
            response_length=128,
            bsz=1
        )
        
        # Basic assertions
        assert isinstance(latencies, list)
        assert len(latencies) == 129  # prompt_phase + response_length
        assert all(isinstance(x, float) for x in latencies)
        assert alpha > 0
        assert beta > 0
        assert c > 0

def test_file_not_found_gpu_info(setup_mock_files):
    # Delete the gpu.json file to test file not found case
    os.remove(os.path.join(setup_mock_files, "data", "gpu.json"))
    
    with patch("simulate.PROJECT_ROOT_PATH", setup_mock_files):
        with pytest.raises(Exception) as exc_info:
            get_gpu_info("NVIDIA A100-SXM4-40GB")
        assert "Could not find gpu information" in str(exc_info.value)

def test_file_not_found_model_info(setup_mock_files):
    # Delete the model.json file to test file not found case
    os.remove(os.path.join(setup_mock_files, "data", "model.json"))
    
    with patch("simulate.PROJECT_ROOT_PATH", setup_mock_files):
        with pytest.raises(Exception) as exc_info:
            get_model_info("llama-8b")
        assert "Could not find model information" in str(exc_info.value)

def test_file_not_found_ptps(setup_mock_files):
    # Delete the ptps.json file to test file not found case
    os.remove(os.path.join(setup_mock_files, "data", "ptps.json"))
    
    with patch("simulate.PROJECT_ROOT_PATH", setup_mock_files):
        with pytest.raises(Exception) as exc_info:
            get_ptps("llama-8b", "NVIDIA A100-SXM4-40GB")
        assert "Could not find ptp" in str(exc_info.value)