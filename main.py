from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from utils import SHORT_GPU_NAME_DEVICE_CUDA_NAME_MAP, PROJECT_ROOT_PATH
import numpy as np
from simulate import simulate

app = FastAPI()

class ModelRequest(BaseModel):
    model_name: str = Field(..., description="Specify the model name")
    gpu_name: str = Field(..., description="Specify the GPU name")
    prompt_length: int = Field(1024, description="Number of tokens in prompt")
    response_length: int = Field(128, description="Number of tokens in response")
    bsz: int = Field(1, description="Batch size")
    tp_level: int = Field(1, description="Tensor Parallelism Level")

@app.post("/simulate")
async def simulate_latency(request: ModelRequest):
    # Simulate processing
    cuda_device_name = SHORT_GPU_NAME_DEVICE_CUDA_NAME_MAP.get(request.gpu_name)
    if cuda_device_name == None:
        print(f"gpu name: {request.gpu_name} not supported!")
        exit(-1)
    latencys, _,_,_ = simulate(request.model_name, cuda_device_name, request.prompt_length, request.response_length, bsz=request.bsz, tp_level=request.tp_level)
    ttft = latencys[0]
    avg_tbt = np.mean(latencys[1:])
    e2e_latency = sum(latencys)
    return {
        "ttft": ttft,
        "avg_tbt": avg_tbt,
        "e2e_latency": e2e_latency,
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)