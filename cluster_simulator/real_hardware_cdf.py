import matplotlib.pyplot as plt
from dateutil import parser
from zoneinfo import ZoneInfo
import numpy as np
import json
from typing import Tuple
from tqdm import tqdm
from common import TimeSeriesFunction, EventTimestamp

entry = []

with open("/home/siyu/Documents/data_process/logs/logs_2025-05-27-16-04-01/run_1/loadgen_result.json", "r") as f:
    for line_number, line in enumerate(f.readlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            request_entry = json.loads(line)
            entry.append(request_entry)
        except json.JSONDecodeError as e:
            print(f"Error parsing line {line_number}: {e}")
            print(f"Problematic line content: {line[:100]}...") 

print(entry[1].keys())


def sg_time_to_utc(sg_time):
    return sg_time.astimezone(ZoneInfo("UTC")).isoformat().replace('+00:00', 'Z')

def convert_request_to_event_ts(response) -> Tuple[EventTimestamp, EventTimestamp]:
    print(response.keys())
    actual_send_ts = response["actual_send_time"]
    arrive_ts = response["response"]["response"]["metrics_list"][0]["arrival_time"] # is ms
    
    # arrive_ts = response["response"]["response"]["metrics_list"][0]["arrival_time"] # is ms
    finish_ts = response["response"]["response"]["metrics_list"][0]["finished_time"]
    # print(f"arrive_ts:{arrive_ts}")
    # 2025-05-27T16:11:25.927600234+08:00 to timestamp
    actual_clean_send_ts = parser.isoparse(actual_send_ts).timestamp()
    # print(f"actual_clean_send_ts:{actual_clean_send_ts}")
    router_ts = arrive_ts - actual_clean_send_ts
    # print(f"router:{router_ts}")
    router_event_ts = EventTimestamp(actual_clean_send_ts, event="router")
    arrive_event_ts = EventTimestamp(arrive_ts, event="arrive")
    finish_event_ts = EventTimestamp(finish_ts, event="finish")
    return arrive_event_ts, router_event_ts, router_ts, finish_event_ts

def convert_processed_trace_to_concurrency_series(entry) -> TimeSeriesFunction:
    respond_list = entry
    event_ts_list = []
    router_ts_list = []
    for response in tqdm(respond_list, desc="Converted", unit="req"):
        arrive_ts, router_event_ts, router_ts, finish_ts = convert_request_to_event_ts(response)
        # concurrencys, timestamps = convert_request_to_event_ts(entry)
        
        event_ts_list.append(arrive_ts)
        event_ts_list.append(finish_ts)
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
        elif event_ts.event == "finish":
        # elif event_ts.event == "router":
            concurrency += 1
        concurrencys.append(concurrency)

    print(f"length of concurrencys:{len(concurrencys)}")
    print(f"length of timestamps:{len(timestamps)}")
    concurrency_series = TimeSeriesFunction(timestamps=timestamps, values=concurrencys)
    return concurrency_series, router_ts_list

concurrency_series, router_ts_list = convert_processed_trace_to_concurrency_series(entry)
fig = concurrency_series.plot()
ax = fig.gca()
ax.set_ylabel(f"Real concurrency")
ax.set_xlabel(f"Timestamps")
fig.savefig("../results/real_concurrency.png")


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
    

# plot_router_ts_cdf(router_ts_list)
