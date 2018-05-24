import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import pickle

x_range = 10.0
y_range = 10.0
TEST_TIMES = 7
TEST_PER_IMAGE = 1
IMAGES = 1000
NODES = 20
DISTANCE = 4.1
NOISE = 0.05
BATCH_SIZE = 20
LEARNING_RATE = 0.01
EPOCH = 20
optimizer = tf.train.AdamOptimizer

config = tf.ConfigProto(device_count={'GPU': 0})
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

USED_NODES = 15


class Master(object):
    def __init__(self):

        self.nodes = np.loadtxt('point.data')
        self.distances = np.loadtxt('distance.data')
        self.hops = np.loadtxt('hop.data')

        self.nodes = self.nodes.reshape(IMAGES, NODES, 2)
        self.hops = self.hops.reshape(IMAGES, NODES, NODES)

        self.distances = self.distances.reshape(IMAGES, NODES, NODES)
        noise = np.random.normal(0, 0.05, size=self.distances.shape)
        self.distances *= (1 + noise)

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

        m = Model()

        for i in range(1000):
            if i in self.blacklist:
                continue
            X = np.zeros((20, 3 * USED_NODES))
            y = np.zeros((20, 2))
            for j in range(20):
                for h in range(20):
                    self.distances[h] *= 0.95 ** (self.hops[i, j, h] - 1)

                index_ = list(range(20))
                index_.remove(j)
                index = list(np.argsort(self.hops[i][j])[:USED_NODES])

                dis = np.array(self.distances[i, j, index])
                X[j] = np.array(
                    [self.nodes[i, index, 0], self.nodes[i, index, 1], dis]).flatten()
                y[j] = np.array([self.nodes[i, j]]).flatten()
            m.train(X, y)

        losses = []
        for i in range(500, 1000):
            if i in self.blacklist:
                continue
            X = np.zeros((20, 3 * USED_NODES))
            y = np.zeros((20, 2))
            for j in range(20):
                for h in range(20):
                    self.distances[h] *= 0.95 ** (self.hops[i, j, h] - 1)

                index_ = list(range(20))
                index_.remove(j)
                index = list(np.argsort(self.hops[i][j])[:USED_NODES])

                dis = np.array(self.distances[i, j, index])
                X[j] = np.array(
                    [self.nodes[i, index, 0], self.nodes[i, index, 1], dis]).flatten()
                y[j] = np.array([self.nodes[i, j]]).flatten()
            loss, y, pos = m.test(X, y)
            losses.append(loss)
            #plot(y, pos, self.hops[i], i)
        m.save()
        print(np.mean(losses))


class Model(object):
    def __init__(self):
        global optimizer
        self.sess = tf.Session(config=config)

        self.x = tf.placeholder(
            tf.float32, shape=[BATCH_SIZE, 3 * USED_NODES])
        self.self_pos = tf.placeholder(
            tf.float32, shape=[BATCH_SIZE, 2])
        activation = tf.nn.sigmoid
        dense1 = tf.layers.dense(
            self.x, 3 * USED_NODES, activation=activation)
        dense4 = tf.layers.dense(
            dense1, 3 * USED_NODES, activation=activation)
        dense4_self_pos = tf.concat([dense4, self.self_pos], 1)
        self.pos = tf.layers.dense(dense4_self_pos, 2)

        self.y = tf.placeholder(tf.float32, [BATCH_SIZE, 2])
        self.loss = tf.reduce_sum(tf.sqrt(
            (self.pos[:, 0] - self.y[:, 0]) ** 2 + (self.pos[:, 1] - self.y[:, 1]) ** 2))
        self.optimizer = optimizer(learning_rate=LEARNING_RATE)
        self.train_step = self.optimizer.minimize(self.loss)
        tf.global_variables_initializer().run(session=self.sess)

    def train(self, X, y):
        loss = 0.0
        # print(np.array(self_pos).shape)
        with self.sess.as_default():
            for i in range(EPOCH):
                loss, _ = tf.get_default_session().run(
                    [
                        self.loss, self.train_step
                    ],
                    feed_dict={
                        self.x: X,
                        self.self_pos: np.zeros((20, 2), dtype=np.float),
                        self.y: y
                    })
        # print(loss)

    def test(self, X, y):
        with self.sess.as_default():
            loss, pos = tf.get_default_session().run(
                [
                    self.loss, self.pos
                ],
                feed_dict={
                    self.x: X,
                    self.self_pos: np.zeros((20, 2), dtype=np.float),
                    self.y: y
                })
            print(loss)
            # print(np.mean((pos - y).flatten()))
            return loss, y, pos

    def save(self):
        tvars = tf.trainable_variables()
        tvars_vals = self.sess.run(tvars)

        result = {}
        for var, val in zip(tvars, tvars_vals):
            result[var.name] = val
        pickle.dump(result, open('variables.pkl', 'wb'))

        # saver = tf.train.Saver()
        # saver.save(self.sess, 'Model/model.ckpt')


def plot(y, pos, hops, image_num):
    fig, ax = plt.subplots()

    ax.scatter(
        y[:, 0],
        y[:, 1],
    )
    ax.scatter(y[:, 0], y[:, 1])

    for i in range(20):
        for j in range(20):
            if hops[i, j] == 1:
                ax.plot(y[[i, j], 0],
                        y[[i, j], 1])

    for i in range(len(pos)):
        ax.annotate(str(i), (pos[i, 0], pos[i, 1]))
        ax.annotate(str(i), (y[i, 0], y[i, 1]))
    fp = str(image_num) + '.png'
    fig.savefig(os.path.join('images', fp))
    # plt.show()


if __name__ == '__main__':
    Master()
