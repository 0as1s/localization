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

scores = np.array([list(scores1.values()), list(scores2.values())])
scores = scores.T
print(len(list(filter(lambda x: x[0] < x[1], scores))))
