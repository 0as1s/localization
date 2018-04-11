# encoding=utf-8
import numpy as np
import json
from trainer import Trainer
from multiprocessing import Pool, cpu_count
from datetime import datetime


# batch_size = 20
x_range = 10
y_range = 10
TEST_TIMES = 4
TEST_PER_IMAGE = 2


class Master(object):
    def __init__(self):

        self.nodes = np.loadtxt('point.data')
        self.distances = np.loadtxt('distance.data')
        self.hops = np.loadtxt('hop.data')

        self.nodes = self.nodes.reshape(100, 20, 2)
        self.hops = self.hops.reshape(100, 20, 20)
        self.distances = self.distances.reshape(100, 20, 20)

        self.blacklist = []
        for i, m in enumerate(self.hops):
            if any(map(lambda x: x == -1, m.flatten())):
                self.blacklist.append(i)
        self.result = {}

    def run(self, i=None, beacon_index=None):

        # 每张图传递给每个trainer
        if not i:
            i = np.random.randint(len(self.distances))
            while(i in self.blacklist):
                i = np.random.randint(len(self.distances))

        print(i)
        if not beacon_index:
            beacon_index = sorted(np.random.choice(
                len(self.distances[i]), 3, replace=False))

            while(True):
                beacon_index = sorted(np.random.choice(
                    len(self.distances[i]), 3, replace=False))
                xs = self.nodes[i, beacon_index, 0]
                ys = self.nodes[i, beacon_index, 1]
                if not 0.5 < (((ys[2]-ys[1]) / (xs[2]-xs[1]))/((ys[1] - ys[0])/(xs[1]-xs[0]))) < 1.5:
                    break

        print(beacon_index)
        beacons = self.nodes[i][beacon_index]
        trainer = Trainer(
            self.distances[i], self.hops[i], x_range, y_range, beacon_index, beacons, self.nodes[i], i)
        loss = trainer.train()
        # self.result[str(i) + str(beacon_index)] = loss
        fp = str(i)+str(beacon_index)+'.json'
        json.dump(loss, open(fp, 'w'))
        print(loss)
        # print(i)
        # print(beacon_index)


if __name__ == '__main__':
    m = Master()
    # m.run()
    with Pool(cpu_count() // 2) as p:
        results = []
        for t1 in range(TEST_TIMES):
            i = np.random.randint(len(m.distances))
            while(i in m.blacklist):
                i = np.random.randint(len(m.distances))

            for t2 in range(TEST_PER_IMAGE):
                beacon_index = sorted(np.random.choice(
                    len(m.distances[i]), 3, replace=False))

                while(True):
                    beacon_index = sorted(np.random.choice(
                        len(m.distances[i]), 3, replace=False))
                    xs = m.nodes[i, beacon_index, 0]
                    ys = m.nodes[i, beacon_index, 1]
                    if not 0.5 < (((ys[2]-ys[1]) / (xs[2]-xs[1]))/((ys[1] - ys[0])/(xs[1]-xs[0]))) < 1.5:
                        break
                results.append(p.apply_async(m.run, args=(i, beacon_index)))
        for r in results:
            r.wait()
