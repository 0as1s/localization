import os
import pickle
import json
import numpy as np
import sys
from collections import OrderedDict
import matplotlib
matplotlib.use('AGG')
from matplotlib import pyplot as plt

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
print(len(scores))

r = []
for i in os.listdir(sys.argv[1]):
    r.append(pickle.load(open(os.path.join(sys.argv[1], i), 'rb')))
r1 = np.mean(r, axis=0)
plt.plot(range(len(r1)), r1, label='first_line')

r = []
for i in os.listdir(sys.argv[2]):
    r.append(pickle.load(open(os.path.join(sys.argv[2], i), 'rb')))
r2 = np.mean(r, axis=0)
plt.plot(range(len(r2)), r2, label='second_line')

plt.legend()
plt.savefig('compare.png')
