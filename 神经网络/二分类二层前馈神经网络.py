import numpy as np
import pandas as pd


def layer_sizes(x, y, k):
    n_x = x.shape[0]  # 输入层大小,是样本的特征数，n_x=2
    n_h = k  # 隐藏层大小
    n_y = y.shape[0]  # 输出层大小
    return (n_x, n_h, n_y)


def sigmod(x):
    s = 1 / (1 + np.exp(-x))
    return s


def forward_propagation(x, param):
    '''前向传播'''
    W1 = param["W1"]
    b1 = param["b1"]
    W2 = param["W2"]
    b2 = param["b2"]

    Z1 = np.dot(W1, x) + b1
    A1 = np.tanh(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = np.tanh(Z2)

    cache = {"Z1": Z1, "Z2": Z2, "A1": A1, "A2": A2}
    return cache


def backward_propagation(param, cache, x, y):
    '''反向传播'''
    m = x.shape[1]

    W1 = param["W1"]
    W2 = param["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    # 执行反向传播
    dZ2 = A2 - y
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1 / m * np.dot(dZ1, x.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "dW2": dW2, "db1": db1, "db2": db2, }
    return grads


def compute_loss(A2, y, param):
    m = y.shape[1]  # 样本量
    #     计算交叉熵损失
    logprobs = np.multiply(np.log(A2), y) + np.multiply(np.log(1 - A2), 1 - y)
    loss = -1 / m * np.sum(logprobs)
    #     维度压缩
    loss = np.squeeze(loss)
    return loss


def init_parameters(n_x, n_h, n_y):
    W1 = np.random.rand(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.rand(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    param = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, }
    return param


def update_param(param, grads, learning_rate=0.2):
    W1 = param["W1"]
    dW1 = grads["dW1"]
    W1 -= dW1 * learning_rate

    W2 = param["W2"]
    dW2 = grads["dW2"]
    W2 -= dW2 * learning_rate

    b1 = param["b1"]
    db1 = grads["db1"]
    b1 -= db1 * learning_rate

    b2 = param["b2"]
    db2 = grads["db2"]
    b2 -= db2 * learning_rate

    new_param = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, }

    return new_param


def predict(param, x):
    cache = forward_propagation(x, param)
    predictions = (cache["A2"] > 0.5)
    return predictions


def train(x, y, k, iter_num):
    x = np.array(x.T)
    y = np.array(y).reshape(1, -1)
    print(f"输入的维度是{x.shape}，输出y的维度是{y.shape}")
    # 初始化参数
    x_size, h_size, y_size = layer_sizes(x, y, k)
    param = init_parameters(x_size, h_size, y_size)
    costs = []
    # 梯度下降和参数更新循环
    for i in range(iter_num):
        # 前向传播
        cache = forward_propagation(x, param)
        # 计算损失
        loss = compute_loss(cache["A2"], y, param)
        # 计算梯度
        grads = backward_propagation(param, cache, x, y)
        # 更新参数
        param = update_param(param, grads)
        costs.append(loss)
        print(f"Cost after interation {i} :{loss}")
    return costs, param


if __name__ == '__main__':
    # %%
    np.random.seed(123456)

    datadir = r"D:\PythonEx\machinelearningIntro\机器学习实践\神经网络\data\credit.xlsx"
    data = pd.read_excel(datadir)
    x = data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14']]
    y = data[['d']]

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=100, test_size=100, random_state=123456)
    costs, param = train(X_train, y_train, k=4, iter_num=30)
    # %%
    import matplotlib.pyplot as plt

    plt.plot(pd.DataFrame(costs).dropna())
    plt.show()

    # %%
    y_pred = predict(param, X_test.T)

    # %%
    cost = (np.array(y_pred).flatten()) - (np.array(y_test).flatten())
    ce = cost[cost != 0]

    z1 = len(ce)
    z2 = len(y_test)
    print(z1, z2)
    print(z1 / z2)
