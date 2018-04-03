# encoding=utf-8
import tensorflow as tf
import numpy as np

DISTANCE_WEIGHTING = 'ONLY_NEAR'
LEARNING_RATE = 0.001
EPOCH = 10


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
                 true_nodes, j):

        self.true_nodes = true_nodes
        self.beacon_index = beacon_index
        self.index = []
        for i, h in enumerate(hops):
            if h in (1, ):
                self.index.append(i)

        self.index = list(range(len(nodes)))
        self.nodes = np.array(nodes)
        self.distances = np.array(distances)
        self.hops = np.array(hops)
        self.x_range = x_range
        self.y_range = y_range
        self.n_nodes = len(self.nodes)

        self.activation = None
        self.weights, self.input_weights, self.target_index = self.weigting_distances()

        self.dis_to_pred = self.distances[self.target_index]

        self.sess = tf.Session()
        self.x = tf.placeholder(tf.float64, shape=[1, 4 * self.n_nodes])

        self.dense1 = tf.layers.dense(
            self.x, 4 * self.n_nodes, activation=self.activation)
        self.dense2 = tf.layers.dense(
            self.dense1, 4 * self.n_nodes, activation=self.activation)
        self.dense3 = tf.layers.dense(
            self.dense2, 4 * self.n_nodes, activation=self.activation)
        self.dense4 = tf.layers.dense(
            self.dense3, 4 * self.n_nodes, activation=self.activation)

        self.pos = tf.layers.dense(self.dense4, 2)

        self.xs = tf.placeholder(tf.float64, shape=len(self.target_index))
        self.ys = tf.placeholder(tf.float64, shape=len(self.target_index))

        self.pred_distances = tf.sqrt(
            tf.square(self.xs - self.pos[0][0]) +
            tf.square(self.ys - self.pos[0][1]))

        # self.pred_distances = tf.sqrt((
        #     self.xs - self.pos[0][0])**2 + (self.ys - self.pos[0][1])**2)
        self.true_distances = tf.constant(self.dis_to_pred, dtype=tf.float64)

        self.loss = tf.losses.mean_squared_error(
            self.true_distances, self.pred_distances, self.weights)  # 在指定loss时可以指定weights

        # self.loss = tf.reduce_mean(
        #     tf.square(self.true_distances - self.pred_distances))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        self.train_step = self.optimizer.minimize(self.loss)

        self.i = j
        tf.global_variables_initializer().run(session=self.sess)

    def weigting_distances(self):
        goal_weights = []
        input_weights = list(map(lambda x: 1.0 / x, self.hops))
        for i in self.beacon_index:
            if i in self.index:
                input_weights[i] = 3
        index = []
        if DISTANCE_WEIGHTING == "ONLY_NEAR":
            for i, d in enumerate(self.distances):
                if self.hops[i] in (1, ):
                    goal_weights.append(1/self.hops[i])
                    index.append(i)
            for i in self.beacon_index:
                goal_weights.append(3)
                index.append(i)
        goal_weights = list(map(lambda x: x / sum(goal_weights), goal_weights))
        return goal_weights, input_weights, index

    def train_and_update(self, f):
        x_input = np.array([
            self.nodes[:, 0], self.nodes[:, 1], self.distances,
            self.input_weights
        ]).flatten()

        x_input = [
            list(
                map(lambda x: 0.1 if np.isnan(x) or np.isinf(x) or x == 0 else x,
                    x_input))
        ]

        with self.sess.as_default():
            target_nodes = self.nodes[self.target_index]
            for i in range(EPOCH):
                distance1, distance2, loss, pos, _ = tf.get_default_session(
                ).run(
                    [
                        self.true_distances, self.pred_distances, self.loss,
                        self.pos, self.train_step
                    ],
                    feed_dict={
                        self.x: x_input,
                        self.xs: target_nodes[:, 0],
                        self.ys: target_nodes[:, 1]
                    })
        # print(loss)
        true_pos = self.nodes[self.i]
        true_nodes = self.true_nodes[self.target_index]
        false_nodes = self.nodes[self.target_index]
        f.write(str(self.i))
        f.write('\n')
        f.write("%f, %f" % (pos[0][0], pos[0][1]))
        f.write('\n')
        f.write("%f, %f" % (self.true_nodes[self.i][0],
                            self.true_nodes[self.i][1]))
        f.write('\n')
        f.write(str(self.target_index))
        f.write('\n')
        f.write(str(self.nodes[self.target_index, 0]))
        f.write('\n')
        f.write(str(self.true_nodes[self.target_index, 0]))
        f.write('\n')

        f.write(str(self.nodes[self.target_index, 1]))
        f.write('\n')
        f.write(str(self.true_nodes[self.target_index, 1]))
        f.write('\n')

        f.write(str(loss))
        f.write('\n')
        f.write(str(distance1))
        f.write('\n')

        f.write(
            str(
                self.dis(true_pos[0], true_pos[1], true_nodes[:, 0],
                         true_nodes[:, 1])))
        f.write('\n')
        f.write(str(distance2))
        f.write('\n')

        f.write(
            str(
                self.dis(pos[0][0], pos[0][1], false_nodes[:, 0],
                         false_nodes[:, 1])))
        f.write('\n========================\n')
        return pos[0]

    def upgrade_nodes(self, nodes):
        # print(self.nodes.shape)
        self.nodes = nodes[self.index]

    def partial_update(self, i, x, y):
        self.nodes[i][0] = x
        self.nodes[i][1] = y

    def dis(self, x, y, xs, ys):
        return np.sqrt(np.square(xs - x) + np.square(ys - y))
