# encoding=utf-8
import tensorflow as tf
import numpy as np
import os
import pickle


LEARNING_RATE = 0.01
DISCOUNT = 0.95
EPOCH = 20
timesteps = 15
tuning_timesteps = 5
optimizer = tf.train.AdamOptimizer
DISTANCE = 4.1
USED_NODES = 10

config = tf.ConfigProto(device_count={'GPU': 0})
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Model(object):
    def __init__(self, nodes, distances, hops, x_range, y_range, beacon_index,
                 nodes_map, pos, i, kwargs):

        self.using_gradient = False
        self.using_net = kwargs.get('using_net')
        self.i = i
        self.origin_pos = pos
        self.beacon_index = beacon_index

        self.index = list(range(len(nodes)))
        self.nodes = np.array(nodes)
        self.distances = np.array(distances)
        self.hops = np.array(hops)

        for i in range(len(self.distances)):
            self.distances[i] *= (0.95 ** (self.hops[i] - 1))

        self.used_distances = self.distances[:USED_NODES]
        self.used_nodes = self.nodes[:USED_NODES]
        self.used_hops = self.hops[:USED_NODES]

        self.x_range = x_range
        self.y_range = y_range
        self.n_nodes = len(self.nodes)
        self.nodes_map = nodes_map
        self.update_times = 0
        self.sess = tf.Session(config=config)
        self.kwargs = kwargs
        self.network_fine_tuning = False

        if self.using_net:
            self.use_net()
        else:
            self.use_gradient_desecent()

    def use_net(self):

        global optimizer
        self.x = tf.placeholder(tf.float32, shape=[1, 3 * USED_NODES])

        activation = tf.nn.sigmoid
        if 'activation' in self.kwargs.keys():
            activation = self.kwargs['activation']

        weights, self.target_index, discounts = self.weighting_distances(
            only_near=False)

        dis_to_pred = self.distances[self.target_index]

        self.self_pos = tf.placeholder(tf.float32, shape=[1, 2])

        if self.kwargs.get('maxout'):

            dense1 = tf.layers.dense(
                self.x, 3 * USED_NODES)
            maxout1 = tf.contrib.layers.maxout(dense1, 3 * USED_NODES)
            dense4 = tf.layers.dense(
                maxout1, 3 * USED_NODES)
            maxout2 = tf.contrib.layers.maxout(dense4, 3 * USED_NODES)
            dense4_self_pos = tf.concat([maxout2, self.self_pos], 1)

        else:

            dense1 = tf.layers.dense(
                self.x, 3 * USED_NODES, activation=activation)
            dense4 = tf.layers.dense(
                dense1, 3 * USED_NODES, activation=activation)
            dense4_self_pos = tf.concat([dense4, self.self_pos], 1)

        self.pos = tf.layers.dense(dense4_self_pos, 2)

        self.xs = tf.placeholder(tf.float32, shape=len(self.target_index))
        self.ys = tf.placeholder(tf.float32, shape=len(self.target_index))

        pred_distances = tf.sqrt(
            tf.square(self.xs - self.pos[0][0]) +
            tf.square(self.ys - self.pos[0][1])
        )

        true_distances = tf.constant(dis_to_pred, dtype=tf.float32)

        discounts = tf.constant(discounts, dtype=tf.float32)
        discounted_distances = discounts * true_distances

        self.loss = tf.losses.mean_squared_error(
            discounted_distances, pred_distances, weights
        )

        if self.kwargs.get('l1_regular'):
            l1_regularizer = tf.contrib.layers.l1_regularizer(
                scale=0.005, scope=None
            )
            weights = tf.trainable_variables()  # all vars of your graph
            regularization_penalty = tf.contrib.layers.apply_regularization(
                l1_regularizer, weights)

            self.loss = self.loss + regularization_penalty

        self.optimizer = optimizer(learning_rate=LEARNING_RATE)
        self.train_step = self.optimizer.minimize(self.loss)

        if self.kwargs.get('always_net'):
            weights, self.target_index2, discounts = self.weighting_distances(
                only_near=True)
            dis_to_pred = self.distances[self.target_index2]
            self.xs2 = tf.placeholder(
                tf.float32, shape=len(self.target_index2))
            self.ys2 = tf.placeholder(
                tf.float32, shape=len(self.target_index2))
            pred_distances = tf.sqrt(
                tf.square(self.xs2 - self.pos[0][0]) +
                tf.square(self.ys2 - self.pos[0][1])
            )

            true_distances = tf.constant(dis_to_pred, dtype=tf.float32)

            discounts = tf.constant(discounts, dtype=tf.float32)
            discounted_distances = discounts * true_distances

            self.loss2 = tf.losses.mean_squared_error(
                discounted_distances, pred_distances, weights
            )

            self.train_step2 = self.optimizer.minimize(self.loss2)

        tf.global_variables_initializer().run(session=self.sess)
        if self.kwargs.get('transfer'):
            if len(self.used_distances) == USED_NODES:
                result = pickle.load(open('variables.pkl', 'rb'))
                tvars = tf.trainable_variables()

                for var in tvars:
                    if var in result.keys():
                        self.sess.run(var.assign(result[var]))

    def use_gradient_desecent(self):
        global optimizer

        self.using_gradient = True
        if self.update_times == 0:
            weights, self.target_index, discounts = self.weighting_distances(
                only_near=False)
        else:
            weights, self.target_index, discounts = self.weighting_distances(
                only_near=True)

        dis_to_pred = self.distances[self.target_index]

        self.pos = tf.Variable(self.origin_pos, dtype=tf.float32)
        self.xs = tf.placeholder(tf.float32, shape=len(self.target_index))
        self.ys = tf.placeholder(tf.float32, shape=len(self.target_index))

        pred_distances = tf.sqrt(
            tf.square(self.xs - self.pos[0]) +
            tf.square(self.ys - self.pos[1]))

        true_distances = tf.constant(dis_to_pred, dtype=tf.float32)

        discounted_distances = discounts * true_distances

        self.loss = tf.losses.mean_squared_error(
            discounted_distances, pred_distances, weights
        )

        # self.loss = tf.losses.mean_squared_error(
        #     tf.square(self.discounted_distances), tf.square(
        #         self.pred_distances), self.weights
        # )
        self.optimizer = optimizer(learning_rate=LEARNING_RATE*2)

        self.train_step = self.optimizer.minimize(self.loss)
        tf.global_variables_initializer().run(session=self.sess)

    def weighting_distances(self, only_near=False):
        if self.kwargs.get('low_decay'):
            dis_decay = 0.2
            beacon_decay = 0
        else:
            dis_decay = 0.4
            beacon_decay = 0.2
        goal_weights = []
        index = []

        if self.kwargs.get('only_near'):
            hops_limit = (1,)
        else:
            hops_limit = (1, 2, 3)

        if not only_near:
            for i, d in enumerate(self.distances):
                if self.hops[i] in hops_limit:
                    goal_weights.append((1-dis_decay)**(self.hops[i] - 1))
                    index.append(i)
            for i in self.beacon_index:
                goal_weights.append((1-beacon_decay)**(self.hops[i] - 1))
                index.append(i)

        else:
            for i, d in enumerate(self.distances):
                if self.hops[i] in (1, ):
                    goal_weights.append((1-beacon_decay)**(self.hops[i] - 1))
                    index.append(i)

        goal_weights = list(map(lambda x: x / sum(goal_weights), goal_weights))
        discounts = []
        for i in index:
            discounts.append(DISCOUNT**(self.hops[i] - 1))
        return goal_weights, index, discounts

    def train_and_update(self):
        x_input = np.array([
            self.used_nodes[:, 0], self.used_nodes[:,
                                                   1], self.used_distances,
        ]).flatten()

        with self.sess.as_default():

            if self.kwargs.get('manage_out_of_range'):
                if self.origin_pos[0] < 0:
                    self.origin_pos[0] = self.x_range
                if self.origin_pos[0] > self.x_range:
                    self.origin_pos[0] = 0
                if self.origin_pos[1] < 0:
                    self.origin_pos[1] = self.y_range
                if self.origin_pos[0] > self.y_range:
                    self.origin_pos[1] = 0
                self.partial_update(
                    self.i, self.origin_pos[0], self.origin_pos[1])

            if self.update_times == timesteps - tuning_timesteps:

                if self.kwargs.get('always_net'):
                    self.network_fine_tuning = True
                else:
                    self.use_gradient_desecent()

            if self.kwargs.get('manage_symmetry'):
                if self.update_times == timesteps - 5:
                    right = 0
                    wrong = 0
                    for i, node in enumerate(self.nodes):
                        hop = self.hops[i]
                        dis = np.sqrt(
                            (node[0] - self.origin_pos[0])**2 + (node[1] - self.origin_pos[1])**2)

                        if dis < DISTANCE:
                            if hop == 1:
                                right += 1
                            else:
                                wrong += 1

                        if hop == 1:
                            if dis < DISTANCE:
                                right += 1
                            else:
                                wrong += 1

                    if wrong > right:
                        if self.kwargs.get('cluster_symmetry'):
                            self.origin_pos[0], self.origin_pos[1] = self.find_symmetry(
                                self.origin_pos[0], self.origin_pos[1])
                        else:
                            self.origin_pos[0] = self.x_range - \
                                self.origin_pos[0]
                            self.origin_pos[1] = self.y_range - \
                                self.origin_pos[1]
                        self.partial_update(
                            self.i, self.origin_pos[0], self.origin_pos[1])
                    if not self.using_gradient:
                        self.use_gradient_desecent()

            target_nodes = self.nodes[self.target_index]
            if self.network_fine_tuning:
                target_nodes = self.nodes[self.target_index2]

            if not self.using_gradient:
                if self.network_fine_tuning:
                    for i in range(EPOCH):
                        loss, pos, _ = tf.get_default_session().run(
                            [
                                self.loss2, self.pos, self.train_step2
                            ],
                            feed_dict={
                                self.x: [x_input],
                                self.xs2: target_nodes[:, 0],
                                self.ys2: target_nodes[:, 1],
                                self.self_pos: [self.origin_pos],
                            })
                        self.origin_pos[0] = min(
                            self.x_range, max(self.origin_pos[0], 0.0))
                        self.origin_pos[1] = min(
                            self.y_range, max(self.origin_pos[1], 0.0))
                        self.partial_update(
                            self.i, self.origin_pos[0], self.origin_pos[1])
                        x_input = np.array([
                            self.nodes[:, 0], self.nodes[:, 1], self.distances,
                        ]).flatten()

                else:
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
                        self.origin_pos[0] = min(
                            self.x_range, max(self.origin_pos[0], 0.0))
                        self.origin_pos[1] = min(
                            self.y_range, max(self.origin_pos[1], 0.0))
                        self.partial_update(
                            self.i, self.origin_pos[0], self.origin_pos[1])
                        x_input = np.array([
                            self.used_nodes[:, 0], self.used_nodes[:,
                                                                   1], self.used_distances,
                        ]).flatten()

            else:
                for i in range(EPOCH):
                    loss, pos, _ = tf.get_default_session().run(
                        [
                            self.loss, self.pos, self.train_step
                        ],
                        feed_dict={
                            self.xs: target_nodes[:, 0],
                            self.ys: target_nodes[:, 1],
                        })

        self.update_times += 1
        if self.using_gradient:
            self.origin_pos = pos
        else:
            self.origin_pos = pos[0]

        self.origin_pos[0] = min(self.x_range, max(self.origin_pos[0], 0.0))
        self.origin_pos[1] = min(self.y_range, max(self.origin_pos[1], 0.0))

        return self.origin_pos, loss

    def partial_update(self, i, x, y):
        if i in self.nodes_map.keys():
            self.nodes[self.nodes_map[i]][0] = x
            self.nodes[self.nodes_map[i]][1] = y

    def find_symmetry(self, x, y):
        index = self.hops[self.hops == 1]
        if index is None:
            return x, y
        nodes = self.nodes[index]
        center_x = np.mean(nodes[:, 0])
        center_y = np.mean(nodes[:, 1])
        return center_x + (center_x - x), center_y + (center_y - y)


class Tester(object):
    def __init__(self):
        self.sess = tf.Session(config=config)
        self.x = tf.placeholder(tf.float32, shape=[1, 3 * USED_NODES])

        activation = tf.nn.sigmoid

        dense1 = tf.layers.dense(
            self.x, 3 * USED_NODES, activation=activation)

        # dense2 = tf.layers.dense(
        #    dense1, 3 * USED_NODES, activation=activation)

        # dense3 = tf.layers.dense(
        #    dense2, 3 * USED_NODES, activation=activation)

        dense4 = tf.layers.dense(
            dense1, 3 * USED_NODES, activation=activation)

        self.self_pos = tf.placeholder(tf.float32, shape=[1, 2])
        dense4_self_pos = tf.concat([dense4, self.self_pos], 1)

        self.pos = tf.layers.dense(dense4_self_pos, 2)

        # saver = tf.train.Saver()
        # saver.restore(self.sess, 'Model/model.ckpt')
        # print('load')
        tf.global_variables_initializer().run(session=self.sess)

        tvars = tf.trainable_variables()
        tvars_vals = self.sess.run(
            tvars,
            feed_dict={
                self.x: [np.zeros(3*USED_NODES, dtype=np.float)],
                self.self_pos: [[0.0, 0.0]],
            }
        )

        result = {}
        for var, val in zip(tvars, tvars_vals):
            result[var.name] = val
        pickle.dump(result, open('variables.pkl', 'wb'))

        result = pickle.load(open('variables.pkl', 'rb'))
        tvars = tf.trainable_variables()

        for var in tvars:
            if var in result.keys():
                self.sess.run(var.assign(result[var]))


if __name__ == '__main__':
    Tester()
