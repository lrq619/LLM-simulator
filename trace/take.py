import json

with open("sample_0_all_input1_output1_ALL.json", "r") as f:
    data = json.load(f)

print(data[1]["data"].values())
