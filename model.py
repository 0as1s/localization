# encoding=utf-8
import tensorflow as tf
import numpy as np
import os


DISTANCE_WEIGHTING = 'ONLY_NEAR'
LEARNING_RATE = 0.001
DISCOUNT = 0.95
EPOCH = 50
timesteps = 50

config = tf.ConfigProto(device_count={'GPU': 0})
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Model(object):
    # 几种传入值的方法：
    # 需要连接向量的地方用tf.concat
    # 1、直接平铺输入
    # 2、每个点先局部过几层局部连接的节点，再全连接
    # 3、成对输入
    # 3a、按距离排序后，距离差相近的作为一对
    # 3b、按距离排序后，距离差相远的作为一对
    # 4、更多的节点作为一组，组内全连接，组间局部连接
    # label可以设置为加权的距离或者只有直达的点的距离
    # 对于预测距离与跳数不符的惩罚
    # 把loss作为权重
    def __init__(self, nodes, distances, hops, x_range, y_range, beacon_index,
                 nodes_map, pos, i, using_net=True):

        self.using_gradient = False
        self.i = i
        self.origin_pos = pos
        self.beacon_index = beacon_index

        self.index = list(range(len(nodes)))
        self.nodes = np.array(nodes)
        self.distances = np.array(distances)
        self.hops = np.array(hops)
        self.x_range = x_range
        self.y_range = y_range
        self.n_nodes = len(self.nodes)
        self.nodes_map = nodes_map
        self.update_times = 0

        self.activation = tf.nn.sigmoid
        self.weights, self.input_weights, self.target_index, self.discounts = self.weigting_distances()

        self.dis_to_pred = self.distances[self.target_index]

        self.sess = tf.Session(config=config)
        self.x = tf.placeholder(tf.float32, shape=[1, 4 * self.n_nodes])

        self.dense1 = tf.layers.dense(
            self.x, 4 * self.n_nodes, activation=self.activation)

        self.dense2 = tf.layers.dense(
            self.dense1, 4 * self.n_nodes, activation=self.activation)

        self.dense3 = tf.layers.dense(
            self.dense2, 4 * self.n_nodes, activation=self.activation)

        self.dense4 = tf.layers.dense(
            self.dense3, 4 * self.n_nodes, activation=self.activation)

        self.self_pos = tf.placeholder(tf.float32, shape=[1, 2])
        self.dense4_self_pos = tf.concat([self.dense4, self.self_pos], 1)

        self.pos = tf.layers.dense(self.dense4_self_pos, 2)

        self.xs = tf.placeholder(tf.float32, shape=len(self.target_index))
        self.ys = tf.placeholder(tf.float32, shape=len(self.target_index))

        self.pred_distances = tf.sqrt(
            tf.square(self.xs - self.pos[0][0]) +
            tf.square(self.ys - self.pos[0][1]))

        self.true_distances = tf.constant(self.dis_to_pred, dtype=tf.float32)

        self.discounts = tf.constant(self.discounts, dtype=tf.float32)
        self.discounted_distances = self.discounts * self.true_distances

        # self.loss = tf.losses.mean_squared_error(
        #     self.discounted_distances, self.pred_distances, self.weights
        # )

        self.loss = tf.losses.mean_squared_error(
            tf.square(self.discounted_distances), tf.square(
                self.pred_distances), self.weights
        )

        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        self.train_step = self.optimizer.minimize(self.loss)

        tf.global_variables_initializer().run(session=self.sess)
        if not using_net:
            self.use_gradient_desecent()

    def weigting_distances(self):
        goal_weights = []
        input_weights = list(map(lambda x: 1.0 / (x + 1), self.hops))
        for i in self.beacon_index:
            if i in self.index:
                input_weights[i] = 3
        index = []
        if DISTANCE_WEIGHTING == "ONLY_NEAR":
            for i, d in enumerate(self.distances):
                if self.hops[i] in (1, 2, 3):
                    goal_weights.append(0.4**(self.hops[i] - 1))
                    index.append(i)
            for i in self.beacon_index:
                if self.hops[i] == -1:
                    continue
                goal_weights.append(2 * 0.5**(self.hops[i] - 1))
                index.append(i)
        goal_weights = list(map(lambda x: x / sum(goal_weights), goal_weights))
        discounts = []
        for i in index:
            discounts.append(DISCOUNT**(self.hops[i] - 1))
        return goal_weights, input_weights, index, discounts

    def train_and_update(self):
        x_input = np.array([
            self.nodes[:, 0], self.nodes[:, 1], self.distances,
            self.input_weights
        ]).flatten()

        with self.sess.as_default():
            target_nodes = self.nodes[self.target_index]

            # if self.origin_pos[0] < 0:
            #     self.origin_pos[0] = self.x_range
            # if self.origin_pos[0] > self.x_range:
            #     self.origin_pos[0] = 0
            # if self.origin_pos[1] < 0:
            #     self.origin_pos[1] = self.y_range
            # if self.origin_pos[0] > self.y_range:
            #     self.origin_pos[1] = 0
            # self.partial_update(
            #     self.i, self.origin_pos[0], self.origin_pos[1])

            if self.update_times == timesteps - 1:
                right = 0
                wrong = 0
                for i, node in enumerate(self.nodes):
                    hop = self.hops[i]
                    dis = np.sqrt(
                        (node[0] - self.origin_pos[0])**2 + (node[1] - self.origin_pos[1])**2)

                    if dis < 4.1:
                        if hop == 1:
                            right += 1
                        else:
                            wrong += 1

                    if hop == 1:
                        if dis < 4.1:
                            right += 1
                        else:
                            wrong += 1

                if wrong > right:
                    self.origin_pos[0] = self.x_range - self.origin_pos[0]
                    self.origin_pos[1] = self.y_range - self.origin_pos[1]
                    self.partial_update(
                        self.i, self.origin_pos[0], self.origin_pos[1])
                if not self.using_gradient:
                    self.use_gradient_desecent()

            for i in range(EPOCH):
                loss, pos, _ = tf.get_default_session().run(
                    [
                        self.loss, self.pos, self.train_step
                    ],
                    feed_dict={
                        self.x: [x_input],
                        self.xs: target_nodes[:, 0],
                        self.ys: target_nodes[:, 1],
                        self.self_pos: [self.origin_pos],
                    })

        self.update_times += 1
        if self.using_gradient:
            self.origin_pos = pos
        else:
            self.origin_pos = pos[0]

        self.origin_pos[0] = np.min(
            [np.max([self.origin_pos[0], 0.0]), self.x_range])
        self.origin_pos[1] = np.min(
            [np.max([self.origin_pos[1], 0.0]), self.x_range])

        return self.origin_pos, loss

    def partial_update(self, i, x, y):
        if i in self.nodes_map.keys():
            self.nodes[self.nodes_map[i]][0] = x
            self.nodes[self.nodes_map[i]][1] = y

    def use_gradient_desecent(self):
        self.using_gradient = True
        self.pos = tf.Variable(self.origin_pos, dtype=tf.float32)

        self.xs = tf.placeholder(tf.float32, shape=len(self.target_index))
        self.ys = tf.placeholder(tf.float32, shape=len(self.target_index))

        self.pred_distances = tf.sqrt(
            tf.square(self.xs - self.pos[0]) +
            tf.square(self.ys - self.pos[1]))

        self.true_distances = tf.constant(self.dis_to_pred, dtype=tf.float32)

        self.discounted_distances = self.discounts * self.true_distances

        # self.loss = tf.losses.mean_squared_error(
        #     self.discounted_distances, self.pred_distances, self.weights
        # )

        self.loss = tf.losses.mean_squared_error(
            tf.square(self.discounted_distances), tf.square(
                self.pred_distances), self.weights
        )

        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        self.train_step = self.optimizer.minimize(self.loss)
        tf.global_variables_initializer().run(session=self.sess)
