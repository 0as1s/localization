import os
import json
import numpy as np
from collections import OrderedDict

jsons = filter(lambda x: x.endswith('.1.json'), os.listdir())
scores1 = OrderedDict()
for j in jsons:
    i = int(j.split('[')[0])
    mid = j.split('[')[1].split(']')[0]
    index = json.loads('['+mid+']')
    scores1[str([i, index])] = json.load(open(j, 'r'))
print(np.mean(list(scores1.values())))
print(len(list(filter(lambda x: x[1] > 1, scores1.items())))/len(scores1))

jsons = filter(lambda x: x.endswith('.3.json'), os.listdir())
scores3 = OrderedDict()
for j in jsons:
    i = int(j.split('[')[0])
    mid = j.split('[')[1].split(']')[0]
    index = json.loads('['+mid+']')
    scores3[str([i, index])] = json.load(open(j, 'r'))
print(np.mean(list(scores3.values())))

print('-------------------')

jsons = filter(lambda x: x.endswith('.2.json'), os.listdir())
scores2 = OrderedDict()
for j in jsons:
    i = int(j.split('[')[0])
    mid = j.split('[')[1].split(']')[0]
    index = json.loads('['+mid+']')
    scores2[str([i, index])] = json.load(open(j, 'r'))
print(np.mean(list(scores2.values())))
print(len(list(filter(lambda x: x[1] > 1, scores2.items())))/len(scores2))

jsons = filter(lambda x: x.endswith('.4.json'), os.listdir())
scores4 = OrderedDict()
for j in jsons:
    i = int(j.split('[')[0])
    mid = j.split('[')[1].split(']')[0]
    index = json.loads('['+mid+']')
    scores4[str([i, index])] = json.load(open(j, 'r'))
print(np.mean(list(scores4.values())))

print('-------------------')

scores = np.array(list(zip(list(scores1.values()), list(scores3.values()))))
print(len(list(filter(lambda x: x[0] - x[1] < 0.01, scores)))/len(scores))

print(len(list(filter(lambda x: x[1] - x[0] > 0.1, scores)))/len(scores))

scores = np.array(list(zip(list(scores2.values()), list(scores4.values()))))
print(len(list(filter(lambda x: x[0] <= x[1], scores)))/len(scores))
print('=====================')
