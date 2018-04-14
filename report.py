import os
import json
jsons = filter(lambda x: x.endswith('json'), os.listdir())
scores = {}
for j in jsons:
    i = int(j.split('[')[0])
    mid = j.split('[')[1].split(']')[0]
    index = json.loads('['+mid+']')
    scores[str([i, index])] = json.load(open(j, 'r'))
json.dump(scores, open('result.json', 'w'))
