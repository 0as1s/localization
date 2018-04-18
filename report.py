import os
import json
import numpy as np
jsons = filter(lambda x: x.endswith('.1.json'), os.listdir())
scores = {}
for j in jsons:
    i = int(j.split('[')[0])
    mid = j.split('[')[1].split(']')[0]
    index = json.loads('['+mid+']')
    scores[str([i, index])] = json.load(open(j, 'r'))
print(np.mean(list(scores.values())))
print(len(list(filter(lambda x: x[1]>1, scores.items())))/len(scores))
print('-------------------')
jsons = filter(lambda x: x.endswith('.2.json'), os.listdir())
scores = {}
for j in jsons:
    i = int(j.split('[')[0])
    mid = j.split('[')[1].split(']')[0]
    index = json.loads('['+mid+']')
    scores[str([i, index])] = json.load(open(j, 'r'))
print(np.mean(list(scores.values())))
print(len(list(filter(lambda x: x[1]>1, scores.items())))/len(scores))
