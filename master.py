# encoding=utf-8
import numpy as np
from trainer import Trainer


# batch_size = 20
x_range = 10
y_range = 10


class Master(object):
    def __init__(self):
        # self.nodes = np.load('nodes.npy')
        # self.distances = np.load('distances.npy')
        # self.hops = np.load('hops.npy')

        self.nodes = np.loadtxt('point.data')
        self.distances = np.loadtxt('distance.data')
        self.hops = np.loadtxt('hop.data')

        self.nodes = self.nodes.reshape(100, 20, 2)
        self.hops = self.hops.reshape(100, 20, 20)
        self.distances = self.distances.reshape(100, 20, 20)

    def run(self):

        # 每张图传递给每个trainer
        i = np.random.randint(len(self.distances))
        i = 43
        print(i)
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
        beacon_index = [10, 15, 16]
        beacons = self.nodes[i][beacon_index]
        trainer = Trainer(
            self.distances[i], self.hops[i], x_range, y_range, beacon_index, beacons, self.nodes[i])
        trainer.train()
        print(i)
        print(beacon_index)


if __name__ == '__main__':
    m = Master()
    m.run()
