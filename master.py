# encoding=utf-8
import numpy as np
import json
from trainer import Trainer
from multiprocessing import Pool, cpu_count


# batch_size = 20
x_range = 10.0
y_range = 10.0
TEST_TIMES = 4
TEST_PER_IMAGE = 1

# setting:
# using_net
# manage_out_of_range
# pre_train
# only_near
# using_net
# cluster_symmetry
# manage_symmetry
# l1_regular
# mean_pos
# activation: None, tf.nn.sigmoid, tf.nn.tanh

settings1 = {
    'manage_out_of_range': True,
    'pre_train': True,
    'using_net': True,
    # 'mean_pos': True,
    'manage_symmetry': True,
}

settings2 = {
    'manage_out_of_range': True,
    'pre_train': True,
    # 'mean_pos': True,
    'manage_symmetry': True,
}


class Master(object):
    def __init__(self):

        self.nodes = np.loadtxt('point.data')
        self.distances = np.loadtxt('distance.data')
        self.hops = np.loadtxt('hop.data')

        self.nodes = self.nodes.reshape(1000, 20, 2)
        self.hops = self.hops.reshape(1000, 20, 20)
        self.distances = self.distances.reshape(1000, 20, 20)

        self.blacklist = []
        for i, m in enumerate(self.hops):
            if any(map(lambda x: x == -1, m.flatten())):
                self.blacklist.append(i)
            for hops in m:
                count = 0
                for h in hops:
                    if h == 1:
                        count += 1
                if count <= 2:
                    self.blacklist.append(i)
                    break

    def run(self, i=None, beacon_index=None):

        # 每张图传递给每个trainer
        # i = 16
        print(i)
        # beacon_index = [4, 11, 15]
        print(beacon_index)
        beacons = self.nodes[i][beacon_index]
        trainer = Trainer(
            self.distances[i], self.hops[i], x_range, y_range, beacon_index, beacons, self.nodes[i], i, settings1)
        loss, loss3, nodes = trainer.train()
        # self.result[str(i) + str(beacon_index)] = loss
        fp = str(i)+str(beacon_index)+'.1.json'
        json.dump(loss, open(fp, 'w'))
        fp = str(i)+str(beacon_index)+'.3.json'
        json.dump(loss3, open(fp, 'w'))
        fp = str(i)+str(beacon_index)+'.1.npy'
        np.save(fp, nodes)

        trainer = Trainer(
            self.distances[i], self.hops[i], x_range, y_range, beacon_index, beacons, self.nodes[i], i, settings2)
        loss2, loss4, nodes = trainer.train()
        # self.result[str(i) + str(beacon_index)] = loss
        fp = str(i)+str(beacon_index)+'.2.json'
        json.dump(loss2, open(fp, 'w'))
        fp = str(i)+str(beacon_index)+'.4.json'
        json.dump(loss4, open(fp, 'w'))
        fp = str(i)+str(beacon_index)+'.2.npy'
        np.save(fp, nodes)

        return loss, loss2
        # print(loss)
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

                    k1 = (ys[1] - ys[0])/(xs[1]-xs[0])
                    k2 = (ys[2] - ys[0])/(xs[2]-xs[0])
                    k3 = (ys[2] - ys[1])/(xs[2]-xs[1])

                    flag1 = not (0.6 < abs(k1/k2) < 1.4)
                    flag2 = not (0.6 < abs(k1/k3) < 1.4)
                    flag3 = not (0.6 < abs(k2/k3) < 1.4)

                    flag4 = m.distances[i][beacon_index[0]
                                           ][beacon_index[1]] > 3
                    flag5 = m.distances[i][beacon_index[1]
                                           ][beacon_index[2]] > 3
                    flag6 = m.distances[i][beacon_index[0]
                                           ][beacon_index[2]] > 3
                    if flag1 and flag2 and flag3 and flag4 and flag5 and flag6:
                        break
                results.append(p.apply_async(m.run, args=(i, beacon_index)))
        # for r in results:
        #     r.wait()
        for r in results:
            print(r.get())
