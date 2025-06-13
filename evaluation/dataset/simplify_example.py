import json

session_count = 5

with open('evaluation/dataset/locomo_example.json', 'r') as f:
    data = json.load(f)

for item in data:
    new_qa = []
    for qa in item['qa']:
        flag = True
        for evidence in qa['evidence']:
            # split the evidence by ':'
            evidence = evidence.split(':')[0]
            # remove the first character
            evidence = evidence[1:]
            evidence_count = int(evidence)
            if evidence_count > session_count:
                flag = False
                break
        if flag:
            new_qa.append(qa)
    data[0]['qa'] = new_qa
    to_delete_keys = []
    for key in item['conversation'].keys():
        if key in ['speaker_a', 'speaker_b'] or "date" in key or "timestamp" in key:
            continue
        # skip "session_"
        session_id = key[8:]
        session_id = int(session_id)
        if session_id > session_count:
            to_delete_keys.append(key)
    for key in to_delete_keys:
        del item['conversation'][key]
    data[0]['conversation'] = item['conversation']
    

with open('evaluation/dataset/locomo_example_simplified.json', 'w') as f:
    json.dump(data, f, indent=4)
    