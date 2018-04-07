# encoding=utf-8
import numpy as np

from model import Model, DISCOUNT
from matplotlib import pyplot as plt
from sympy import symbol, solve

timesteps = 50
hops_limit = 3


# 可以比较离线训练+在线测试/在线训练+在线测试/离线训练+迁移学习+在线测试
# 考虑加入降噪层
class Trainer(object):
    def __init__(self, distances, hops, x_range, y_range, beacon_index,
                 beacons, nodes):

        self.distances = distances
        self.beacon_index = list(beacon_index)
        self.beacons = beacons
        self.n_nodes = len(distances)
        self.x_range = x_range
        self.y_range = y_range
        self.nodes = []
        self.true_nodes = nodes

        self.models = []
        self.hops = hops

        self.initialize_nodes()
        self.initialize_models()

    def initialize_models(self):
        nodes_index = list(range(self.n_nodes))
        for i in self.beacon_index:
            nodes_index.remove(i)
        hops = np.copy(self.hops)
        hops[hops == -1] = 999
        nodes_index = np.argsort(np.sum(hops[:, self.beacon_index], axis=1))
        for i in range(self.n_nodes):
            if i in nodes_index:
                sorted_index = []
                # 建立一个全局nodes与单个node需要的nodes的映射
                nodes_map = {}
                for hops in range(1, hops_limit + 1):
                    for j, h in enumerate(self.hops[i]):
                        if h == hops:
                            nodes_map[j] = len(sorted_index)
                            sorted_index.append(j)

                hops = list(self.hops[i, sorted_index])
                dis = list(self.distances[i, sorted_index])
                nodes = list(self.nodes[sorted_index])
                # 将beacon放在最后三个点
                for b in self.beacon_index:
                    hops.append(self.hops[i][b])
                    dis.append(self.distances[i][b])
                    nodes.append(self.beacons[self.beacon_index.index(b)])
                hops = np.array(hops)
                dis = np.array(dis)
                nodes = np.array(nodes)

                beacon_index = [len(sorted_index) + i for i in range(0, 3)]
                self.models.append(Model(nodes, dis, hops, self.x_range, self.y_range, beacon_index, nodes_map))

            else:
                self.models.append(None)

    # 随机生成，后续可以优化为先pre_train，可能会更容易收敛到解
    def initialize_nodes(self):
        xs = self.beacons[:, 0]
        ys = self.beacons[:, 1]
        self.nodes = np.random.random((self.n_nodes, 2))
        self.nodes[:, 0] *= self.x_range
        self.nodes[:, 1] *= self.y_range
        for i in range(self.n_nodes):
            hops = self.hops[i, self.beacon_index]
            ds = self.distances[i, self.beacon_index]
            ds = [ds[j]*(DISCOUNT**(hops[j]-1)) for j in range(len(ds))]
            if i in self.beacon_index:
                self.nodes[i] = self.beacons[self.beacon_index.index(i)]
            else:
                self.nodes[i, 0], self.nodes[i, 1] = self.pos(xs, ys, ds)
        for i in range(len(self.beacons)):
            self.nodes[self.beacon_index[i]] = self.beacons[i]

    # 是否可以让n_nodes个model共享一部分权值, 比如后面的全连接层
    # 如果要共享权值，怎么解决权值在节点间交换的问题
    def train(self):
        f = open('input_x.txt', 'w')
        f2 = open('loss.txt', 'w')
        for t in range(timesteps):
            # 这里如果是个python原生的数组的话，传入的是值，但是如果是一个numpy数组的话，传入的是引用，所以此处需要一个新的数组才缓存值
            new_nodes = np.zeros((self.n_nodes, 2))
            dis_loss = 0
            for i in range(self.n_nodes):
                if i not in self.beacon_index:
                    (x, y), dis_loss = self.models[i].train_and_update(f)
                    for m in self.models:
                        if m:
                            m.partial_update(i, x, y)
                    new_nodes[i][0] = np.min([np.max([x, 0.0]), self.x_range])
                    new_nodes[i][1] = np.min([np.max([y, 0.0]), self.y_range])
                else:
                    new_nodes[i] = self.beacons[self.beacon_index.index(i)]

            self.nodes = new_nodes
            loss = np.mean(np.abs(self.nodes - self.true_nodes))
            print(loss)
            print(dis_loss)
            print("==========")
            # self.plot()
            f2.write(str(loss))
            f.write('\n')
        f.close()
        self.plot()

    def pos(self, xs, ys, ds):
        x1, x2, x3 = list(xs)
        y1, y2, y3 = list(ys)
        d1, d2, d3 = list(ds)
        x = symbol.Symbol('x')
        y = symbol.Symbol('y')
        sol = solve(
            [(x - x1)**2 + (y - y1)**2 - d1**2,
             (x - x2)**2 + (y - y2)**2 - d2**2], [x, y],
            dict=False)
        rx = 0
        ry = 0
        min_loss = 9999
        try:
            for s in sol:
                loss = abs((s[0] - x3)**2 + (s[1] - y3)**2 - d3**2)
                if loss < min_loss:
                    min_loss = loss
                    rx = s[0]
                    ry = s[1]
        except TypeError:
            return np.random.random() * self.x_range, np.random.random(
            ) * self.y_range
        return rx, ry

    def plot(self):
        fig, ax = plt.subplots()

        ax.scatter(
            self.nodes[:, 0],
            self.nodes[:, 1],
        )
        ax.scatter(self.true_nodes[:, 0], self.true_nodes[:, 1])

        s = [20 * 4**2 for n in range(3)]
        ax.scatter(
            self.true_nodes[self.beacon_index, 0],
            self.true_nodes[self.beacon_index, 1],
            color='black',
            s=s)

        for i in range(len(self.nodes)):
            for j in range(i, len(self.nodes)):
                if self.hops[i, j] == 1:
                    ax.plot(self.true_nodes[[i, j], 0],
                            self.true_nodes[[i, j], 1])

        for i in range(len(self.nodes)):
            ax.annotate(str(i), (self.nodes[i, 0], self.nodes[i, 1]))
            ax.annotate(str(i), (self.true_nodes[i, 0], self.true_nodes[i, 1]))
        plt.show()
