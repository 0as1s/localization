# encoding=utf-8
import numpy as np
from trainer import Trainer


# batch_size = 20
x_range = 10
y_range = 10


class Master(object):
    def __init__(self):
        self.nodes = np.load('nodes.npy')
        # rigidities = np.loadtxt('rigidities.txt')
        self.distances = np.load('distances.npy')
        self.hops = np.load('hops.npy')

        # self.rigidities = rigidities.reshape(len(rigidities.flatten()) // n_points, n_points)
        # self.nodes = nodes.reshape(
        #     nodes.shape[0] // batch_size, batch_size, nodes.shape[-1])
        # self.rigidities = rigidities.reshape(
        #     rigidities.shape[0] // batch_size, batch_size,  rigidities.shape[-1]
        # )

    def run(self):

        # 每张图传递给每个trainer
        for i, t in enumerate(self.distances):
            beacon_index = sorted(np.random.choice(len(t), 3, replace=False))
            beacons = self.nodes[i][beacon_index]
            trainer = Trainer(t, self.hops[i], x_range, y_range, beacon_index, beacons, self.nodes[i])
            trainer.train()
            print(beacon_index)
            break


if __name__ == '__main__':
    m = Master()
    m.run()
