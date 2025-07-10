import json

with open("./sample_0_all_input1_output1_ALL-100-1-1.json", "r") as f:
    data = json.load(f)
    
    
timestamps = []
for i in range(len(data)):
    timestamps.append(data[i]["timestamp"])

print(timestamps[-100:])


