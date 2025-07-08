First, compile the sampler:
```
cd LLM-simulator/sampler
go build .
cd ..
```
Then, run cluster simulator with 
```
python -m cluster_simulator.main
```

An example configuration:
```
{
    "gpu_name": "NVIDIA A100-SXM4-40GB",
    "gpu_number": 4,
    "node_number": 4,
    "workloads":[
        {
        "trace_path": "./trace/azure_conv_liquid.json",
        "model": "meta-llama/Meta-Llama-3-8B",
        "sampling_rate": 100,
        "input_length_scale": 1,
        "output_length_scale": 1,
        "tp_level": 1,
        "target": 10
        },
        {
        "trace_path": "./trace/sample_1_all_input1_output1_ALL.json",
        "model": "Qwen/Qwen2.5-32B",
        "sampling_rate": 8,
        "input_length_scale": 1,
        "output_length_scale": 1,
        "tp_level": 4,
        "target": 89
    }
]
}
```


For test
```
python test_autoscaler.py
```