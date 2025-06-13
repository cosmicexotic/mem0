import json
# sample 97 items from longmemeval_oracle.json
with open('evaluation/dataset/longmemeval_oracle.json', 'r') as f:
    data = json.load(f)

seed = 42
# shuffle the data
import random
random.seed(seed)
random.shuffle(data)

# sample 97 items, and save to longmemeval_oracle_sample.json
data = data[:97]
with open('evaluation/dataset/longmemeval_oracle_sample_seed_{}.json'.format(seed), 'w') as f:
    json.dump(data, f, indent=4)