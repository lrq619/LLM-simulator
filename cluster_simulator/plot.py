import matplotlib.pyplot as plt
from datetime import datetime
from dateutil import parser
from zoneinfo import ZoneInfo
import numpy as np
import json
from typing import Tuple, List
from tqdm import tqdm
from common import TimeSeriesFunction, EventTimestamp

with open("../results/alloc_success_number_q_de.json", "r") as f:
    q_de_alloc_success_number = json.load(f)
    q_de_alloc_time = len(q_de_alloc_success_number["values"])
    print(f"Qwen default alloc time: {q_de_alloc_time}")

with open("../results/alloc_success_number_q_kpa.json", "r") as f:
    q_kpa_alloc_success_number = json.load(f)
    q_kpa_alloc_time = len(q_kpa_alloc_success_number["values"])
    print(f"Qwen kpa alloc time: {q_kpa_alloc_time}")

with open("../results/alloc_success_number_lla_de.json", "r") as f:
    lla_de_alloc_success_number = json.load(f)   
    lla_de_alloc_time = len(lla_de_alloc_success_number["values"])
    print(f"LLama default alloc time: {lla_de_alloc_time}")

with open("../results/alloc_success_number_lla_kpa.json", "r") as f:
    lla_kpa_alloc_success_number = json.load(f)
    lla_kpa_alloc_time = len(lla_kpa_alloc_success_number["values"])
    print(f"LLama kpa alloc time: {lla_kpa_alloc_time}")

with open("../results/alloc_success_number_mix_de.json", "r") as f:
    mix_de_alloc_success_number = json.load(f)
    mix_de_alloc_time = len(mix_de_alloc_success_number["values"])
    print(f"Mix default alloc time: {mix_de_alloc_time}")

with open("../results/alloc_success_number_mix_kpa.json", "r") as f:
    mix_kpa_alloc_success_number = json.load(f)
    mix_kpa_alloc_time = len(mix_kpa_alloc_success_number["values"])
    print(f"Mix kpa alloc time: {mix_kpa_alloc_time}")

def default_kpa_cdf(default_alloc_success_number, kpa_alloc_success_number) -> plt.Figure:
    default_values = default_alloc_success_number["values"]
    kpa_values = kpa_alloc_success_number["values"]
    print(f"Avg default #GPU: {np.mean(default_values)}, Avg kpa #GPU: {np.mean(kpa_values)}")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(np.sort(default_values), np.linspace(0, 1, len(default_values), endpoint=False), label=f"GPU number (default)", color="red")
    ax.plot(np.sort(kpa_values), np.linspace(0, 1, len(kpa_values), endpoint=False), label=f"GPU number (kpa)", color="blue")
    ax.set_xlabel(f"GPU number")
    ax.set_ylabel(f"CDF")
    ax.legend()
    ax.grid(True)
    return fig

default_kpa_cdf(q_de_alloc_success_number, q_kpa_alloc_success_number).savefig("../results/qwen_kpa_cdf.png")
default_kpa_cdf(lla_de_alloc_success_number, lla_kpa_alloc_success_number).savefig("../results/lla_kpa_cdf.png")
default_kpa_cdf(mix_de_alloc_success_number, mix_kpa_alloc_success_number).savefig("../results/mix_kpa_cdf.png")


# def plot_alloc_time(alloc_time):
#     fig, ax = plt.subplots(figsize=(8,6))
#     ax.plot(alloc_time, label=f"Alloc time", color="lightblue")
#     ax.set_xlabel(f"Time")
#     ax.set_ylabel(f"Alloc time")
#     ax.legend()
#     return fig

with open("../results/gpu_cost.json", "r") as f:
    gpu_cost = json.load(f)
    
    
def plot_gpu_cost(default_values, kpa_values):
    #plot a bar chart of the gpu cost
    fig, ax = plt.subplots(figsize=(8,6))
    
    # Set the width of each bar and positions
    width = 0.35
    x = np.arange(len(default_values))
    
    # Extract costs and model names
    default_costs = [item["cost"] for item in default_values]
    kpa_costs = [item["cost"] for item in kpa_values]
    model_names = [item["model"] for item in default_values]
    
    # Plot bars side by side by offsetting the x coordinates
    ax.bar(x - width/2, default_costs, width, label=f"Default", color="lightcoral")
    ax.bar(x + width/2, kpa_costs, width, label=f"KPA", color="lightblue")
    
    # Set the tick locations and labels
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_xlabel("Model")
    ax.set_ylabel("Cost")
    ax.legend()
    
    # Add value labels on top of each bar
    for i, v in enumerate(default_costs):
        ax.text(i - width/2, v, f'{v:.2f}', ha='center', va='bottom')
    for i, v in enumerate(kpa_costs):
        ax.text(i + width/2, v, f'{v:.2f}', ha='center', va='bottom')
    
    return fig

   
def plot_alloc_time(default_values, kpa_values):
    #plot a bar chart of the alloc time
    fig, ax = plt.subplots(figsize=(8,6))
    
    # Set the width of each bar and positions
    width = 0.35
    x = np.arange(len(default_values))
    
    # Extract costs and model names
    default_alloc_time = [item["alloc_time"] for item in default_values]
    kpa_alloc_time = [item["alloc_time"] for item in kpa_values]
    model_names = [item["model"] for item in default_values]
    
    # Plot bars side by side by offsetting the x coordinates
    ax.bar(x - width/2, default_alloc_time, width, label=f"Default", color="navajowhite")
    ax.bar(x + width/2, kpa_alloc_time, width, label=f"KPA", color="mediumpurple")
    
    # Set the tick locations and labels
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_xlabel("Model")
    ax.set_ylabel("Alloc time")
    ax.legend()
    
    # Add value labels on top of each bar
    for i, v in enumerate(default_alloc_time):
        ax.text(i - width/2, v, f'{v:.2f}', ha='center', va='bottom')
    for i, v in enumerate(kpa_alloc_time):
        ax.text(i + width/2, v, f'{v:.2f}', ha='center', va='bottom')
    
    return fig

plot_gpu_cost(gpu_cost["default_values"], gpu_cost["kpa_values"]).savefig("../results/gpu_cost.png")
plot_alloc_time(gpu_cost["default_values"], gpu_cost["kpa_values"]).savefig("../results/alloc_time.png")



### plot loadgen result

entry = []
with open("/home/siyu/Documents/logs/logs_2025-05-27-16-04-01_default/run_1/loadgen_result.json", "r") as f:
    for line in f.readlines():
        request_entry = json.loads(line)
        entry.append(request_entry)
print(entry[1].keys())


def sg_time_to_utc(sg_time):
    return sg_time.astimezone(ZoneInfo("UTC")).isoformat().replace('+00:00', 'Z')

def convert_request_to_event_ts(response) -> Tuple[EventTimestamp, EventTimestamp]:
    actual_send_ts = response["actual_send_time"]
    arrive_ts = response["response"]["response"]["metrics_list"][0]["arrival_time"] # is ms
    
    # arrive_ts = response["response"]["response"]["metrics_list"][0]["arrival_time"] # is ms
    # finish_ts = response["response"]["response"]["metrics_list"][0]["finished_time"]
    # print(f"arrive_ts:{arrive_ts}")
    # 2025-05-27T16:11:25.927600234+08:00 to timestamp
    actual_clean_send_ts = parser.isoparse(actual_send_ts).timestamp()
    # print(f"actual_clean_send_ts:{actual_clean_send_ts}")
    router_ts = arrive_ts - actual_clean_send_ts
    # print(f"router:{router_ts}")
    router_event_ts = EventTimestamp(actual_clean_send_ts, event="router")
    arrive_event_ts = EventTimestamp(arrive_ts, event="arrive")
    # finish_event_ts = EventTimestamp(finish_ts, event="finish")
    return arrive_event_ts, router_event_ts, router_ts

def convert_processed_trace_to_concurrency_series(entry) -> TimeSeriesFunction:
    respond_list = entry
    event_ts_list = []
    router_ts_list = []
    for response in tqdm(respond_list, desc="Converted", unit="req"):
        arrive_ts, router_event_ts, router_ts = convert_request_to_event_ts(response)
        # concurrencys, timestamps = convert_request_to_event_ts(entry)
        
        event_ts_list.append(arrive_ts)
        # event_ts_list.append(finish_ts)
        event_ts_list.append(router_event_ts)
        router_ts_list.append(router_ts)
        
    event_ts_list = sorted(event_ts_list,key=lambda event: event.ts)
    # calculate the concurrency
    timestamps = []
    concurrencys = []
    concurrency = 0
    for event_ts in event_ts_list:
        timestamps.append(event_ts.ts)
        if event_ts.event == "arrive":
            concurrency -= 1
        # elif event_ts.event == "finish":
        elif event_ts.event == "router":
            concurrency += 1
        concurrencys.append(concurrency)
        
    print(f"length of concurrencys:{len(concurrencys)}")
    print(f"timestamps:{(timestamps)}")

    concurrency_series = TimeSeriesFunction(timestamps=timestamps, values=concurrencys)
    return concurrency_series, router_ts_list

concurrency_series, router_ts_list = convert_processed_trace_to_concurrency_series(entry)
fig = concurrency_series.plot()
ax = fig.gca()
ax.set_ylabel(f"Router concurrency")
ax.set_xlabel(f"Timestamps")
fig.savefig("../results/router_lifespan.png")


def plot_router_ts_cdf(router_ts_list):
    data = np.sort(router_ts_list)
    cdf = np.arange(1, len(data) + 1) / len(data)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data, cdf, linestyle='-') 
    ax.set_xlabel("Router latency (s)")
    ax.set_ylabel("CDF")
    ax.set_title("Router latency CDF")
    ax.grid(True)
    fig.savefig("../results/router_ts_cdf.png")
    

plot_router_ts_cdf(router_ts_list)
