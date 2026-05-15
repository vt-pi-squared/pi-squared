
import tiktoken
tiktoken_encoder = tiktoken.get_encoding("o200k_base")
count_tokens = lambda text: len(tiktoken_encoder.encode("".join(text), disallowed_special=()))


import json
with open("datasets/ours/v3/ctqar_clean_llm_rt_Qwen3.5-35B-A3B-FP8.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

reasoning_traces_count = [count_tokens(item["llm_generated_reasoning_traces"]) for item in data]
full_messages_count = [count_tokens(item["context"] + item["question"] + str(item["final_answer_sql"]) + item["llm_generated_reasoning_traces"]) for item in data]


import pandas as pd
pd.Series(reasoning_traces_count).describe()
"""
count    13290.000000
mean      1964.435741
std       1326.806431
min        555.000000
25%       1161.000000
50%       1472.000000
75%       2166.750000
max      12575.000000
dtype: float64
"""

pd.Series(full_messages_count).describe()
"""
count     13290.000000
mean      52796.443792
std       27167.618485
min        3115.000000
25%       30870.250000
50%       48048.500000
75%       72898.750000
max      108532.000000
"""

# add two columns to the jsonl file
for item, rt_count, fm_count in zip(data, reasoning_traces_count, full_messages_count):
    item["reasoning_traces_token_count"] = rt_count
    item["full_messages_token_count"] = fm_count

# filter the samples with 
# reasoning traces token count < 2500 and 
# full messages token count < 80000 
filtered_data = [item for item in data if item["reasoning_traces_token_count"] < 1024*2 and item["full_messages_token_count"] < 1024*80]

# save to new jsonl file
with open("datasets/ours/v3/ctqar_clean_llm_rt_Qwen3.5B-A3B-FP8_short_rt_2K_msg_80K.jsonl", "w") as f:
    for item in filtered_data:
        f.write(json.dumps(item) + "\n")

filtered_data_reloaded = [json.loads(line) for line in open("datasets/ours/v3/ctqar_clean_llm_rt_Qwen3.5B-A3B-FP8_short_rt_2K_msg_80K.jsonl", "r")]
for i in range(len(filtered_data_reloaded)):
    if filtered_data_reloaded[i] != filtered_data[i]:
        print(i)
