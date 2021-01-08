import sys,os
sys.path.append(os.pardir) # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
# ニューラルネットワークの層を順番通りに保持するためにOrderDictをインポート
from collections import OrderedDict

# ソフトマックス関数の計算
def softmax(x):
    # オーバーフロー対策のためにxの最大値で引いておく
    x = x - np.max(x, axis=-1, keepdims=True)
    # xのそれぞれの値をxの合計で割った値を返す
    # それぞれの値を確率と見なすことができる
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

# 交差エントロピー誤差の計算
def cross_entropy_error(y, t):
    # ndimで次元数を計算
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    # ミニバッチ学習を行うためにバッチ数を計算
    batch_size = y.shape[0]
    # log(0)を防ぐために1e-7を加える
    # 訓練データの損失関数の和をbatch_sizeで割って正規化したものを計算
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

# SoftmaxWithLossの計算
class SoftmaxWithLoss:
    def __init__(self):
        # 損失
        self.loss = None
        # softmaxの出力
        self.y = None
        # 教師データ
        self.t = None

    # 順伝播
    def forward(self, x, t):
        self.t = t
        # softmax関数の計算
        self.y = softmax(x)
        # 交差エントロピー誤差の計算
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    # 逆伝播
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        # 教師データがone-hot-vectorの場合
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx


# 活性化関数(Reluレイヤ)
# Reluはx<=0において0となり、x>0においてはxをそのまま返す
class Relu:
    # インスタンス変数maskを定義
    # maskはx<=0においてTrue、x>0においてFalse
    def __init__(self):
        self.mask = None

    # 順伝播
    def forward(self, x):
        # x<=0のデータをTrue、x>0のデータをFalseに定義
        self.mask = (x <= 0)
        out = x.copy()
        # self.mask==Trueのデータを0にする
        out[self.mask] = 0

        return out

    # 逆伝播
    def backward(self, dout):
        # self.mask==Trueのデータを0にする
        dout[self.mask] = 0
        dx = dout

        return dx

# Affineレイヤ
class Affine:
    # 初期設定
    def __init__(self, W, b):
        # W,b,xはそれぞれ、重み、バイアス、入力データ
        self.W = W
        self.b = b
        self.x = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    # 順伝播
    def forward(self, x):
        self.x = x

        # ニューロンの重み付き和の計算
        out = np.dot(self.x, self.W) + self.b

        return out

    # 逆伝播
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        # dbの計算、列方向の和を計算
        self.db = np.sum(dout, axis=0)

        return dx


# ニューラルネットワークの実装
class TwoLayerNet:
    # 初期値の設定、input_size,hidden_size,output_size,weight_init_stdはそれぞれ、
    # 入力層のニューロンの数、隠れ層のニューロンの数、出力層のニューロンの数、
    # 重み初期化時のガウス分布のスケール
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重み・バイアスの初期化、W1,b1は1層目の重み、バイアス、W2,b2は2層目の重み、バイアス
        # それぞれの値を辞書に代入
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # レイヤの生成
        # OrderDictによって順番を保持して層を追加できる
        # 1層目にAffineレイヤ、活性化関数にRelu関数、2層目にAffineレイヤ
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        # 最後にSoftmaxWithLossレイヤ
        self.lastLayer = SoftmaxWithLoss()

    # ニューラルネットワークの推論
    def predict(self, x):
        # それぞれのレイヤごとに順伝播を計算する
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # 損失の計算、x:入力データ, t:教師データ
    def loss(self, x, t):
        y = self.predict(x)
        # ニューラルネットワークの最後の層の交差エントロピー誤差の計算
        return self.lastLayer.forward(y, t)

    # ニューラルネットワークの正解率の計算
    def accuracy(self, x, t):
        y = self.predict(x)
        # ニューラルネットワークの出力が最も大きいもののインデックスを取得
        y = np.argmax(y, axis=1)
        # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
        if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x:入力データ, t:教師データ
    def gradient(self, x, t):
        # 順伝播
        self.loss(x, t)

        # 逆伝播
        dout = 1
        # SoftmaxWithLossの逆伝播の計算
        dout = self.lastLayer.backward(dout)

        # 逆伝播を計算するときに層の後ろから計算しないといけないことに注意
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 重み、バイアスの微分を計算
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads


from dataset.mnist import load_mnist

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 先程作成したTwoLayerNetに層ごとのニューロンの数を代入
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 繰り返し回数
iters_num = 10000
# 訓練データの個数(今回は28×28の画像データが60000個）
train_size = x_train.shape[0]
# バッチサイズ
batch_size = 100
# 学習率
learning_rate = 0.1
# 訓練データの損失のリスト
train_loss_list = []
# 訓練データの正解率のリスト
train_acc_list = []
# テストデータの正解率のリスト
test_acc_list = []
# iter_per_epochごとに訓練データ、テストデータの正解率を計算
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # train_size(60000)からbatch_size(100)個の値をランダムに抽出
    batch_mask = np.random.choice(train_size, batch_size)
    # 訓練データからランダムに100個選ぶ
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配の計算
    grad = network.gradient(x_batch, t_batch)

    # 重み、バイアスの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    # 損失の計算
    loss = network.loss(x_batch, t_batch)
    # 損失をリストに追加
    train_loss_list.append(loss)

    # iter_per_epochごとに正解率を追加
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)

#　訓練データの損失のグラフを描画
plt.title('train_loss')
plt.plot([n for n in range(iters_num)],train_loss_list)
plt.xlabel('iters_num')
plt.ylabel('train_loss')
plt.show()

# 訓練データの正解率のグラフを描画
plt.title('train_acc')
plt.plot([n for n in range(0,iters_num,int(iter_per_epoch))],train_acc_list,linestyle='None',marker='.')
plt.xlabel('iters_num')
plt.ylabel('train_acc')
plt.show()

# テストデータの正解率のグラフを描画
plt.title('test_acc')
plt.plot([n for n in range(0,iters_num,int(iter_per_epoch))],test_acc_list,linestyle='None',marker='.')
plt.xlabel('iters_num')
plt.ylabel('test_acc')
plt.show()
