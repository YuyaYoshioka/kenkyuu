import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.random.seed(1234)
tf.set_random_seed(1234)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X_u, u, X_f, layers, lb, ub):

        self.lb = lb
        self.ub = ub

        # 境界条件に使用
        self.x_u = X_u

        # 微分方程式の左辺に使用
        self.x_f = X_f

        self.u = u

        self.layers = layers

        # 重みとバイアスを設定
        self.weights, self.biases = self.initialize_NN(layers)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, 1])

        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, 1])

        # 境界・初期条件における予測値を出力
        self.u_pred = self.net_u(self.x_u_tf)
        # 入力データにおける予測値を出力
        self.f_pred = self.net_f(self.x_f_tf)

        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_pred))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps,
                                                                         # The iteration stops when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol.
                                                                         'gtol': 1e4 * np.finfo(
                                                                             float).eps})  # The iteration will stop when max{|proj g_i | i = 1, ..., n} <= gtol where pg_i is the i-th component of the projected gradient.

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x):
        u = self.neural_net(x, self.weights, self.biases)
        return u

    def net_f(self, x):
        u = self.net_u(x)
        u_x = tf.gradients(u, x)[0]
        f = u - u_x

        return f

    def net_fx(self, x):
        u = self.net_u(x)
        u_x = tf.gradients(u, x)[0]
        u_x_x = tf.gradients(u_x, x)[0]
        f = u_x - u_x_x

        return f

    def callback(self, loss, weights):
        print('Loss:', loss)

    def train(self):

        tf_dict = {self.x_u_tf: self.x_u, self.u_tf: self.u,
                   self.x_f_tf: self.x_f}

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss, self.weights],
                                loss_callback=self.callback)

    def predict(self, X_star):

        u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star})
        f_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star})

        return u_star, f_star


if __name__ == "__main__":
    noise = 0.0

    N_u = 100
    N_f = 10000
    layers = [1, 20, 20, 20, 20, 20, 20, 20, 20, 1]

    X = np.linspace(0, 4, 10000)
    U = np.exp(X) * 10

    X_star = X

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)

    print('lb', lb)
    print('ub', ub)

    X_u_train = np.array([0 for _ in range(N_u)])
    X_u_train = np.vstack(X_u_train)
    u_train = np.array([1 for _ in range(N_u)])
    u_train = np.vstack(u_train)

    X_f_train = lb + (ub - lb) * lhs(1, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))

    X_star = np.vstack(X_star)
    u_star = np.vstack(U)
    print(X_star.shape)
    print(u_star.shape)

    model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub)

    X = np.linspace(0, 5, 10000)
    X_star = X
    X_star = np.vstack(X_star)

    start_time = time.time()
    model.train()
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    u_pred, f_pred = model.predict(X_star)

    plt.plot(X, u_pred)
    plt.show()
    print(f_pred.shape)

    error = 0

    Error = np.abs(u_star - u_pred)
    print(sum)
