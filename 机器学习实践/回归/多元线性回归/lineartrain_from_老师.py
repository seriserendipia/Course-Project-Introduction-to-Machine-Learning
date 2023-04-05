import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def r2_score(y_test, y_pred):
    y_avg = np.mean(y_test)
    ss_tot = np.sum((y_test - y_avg) ** 2)
    ss_res = np.sum((y_test - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def standerlize(x):
    '''标准化（最大最小）'''
    for i in range(len(x)):
        max_min_diff = np.max(x[i]) - np.min(x[i])
        x[i] = np.divide((x[i] - np.min(x[i])), max_min_diff)
        print(x[i])
    print(x)
    return x


# 初始化,w
def init_param(x_dim):
    w = np.ones((1, x_dim)).T
    b = -2
    return w, b


def linear_loss(x, y, w, b):
    num_train = x.shape[0]
    num_feature = x.shape[1]  # TODO num_feature 没有用
    y_hat = np.dot(x, w) + b
    loss = np.sum((y_hat - y) ** 2) / num_train  # 均方损失
    dw = np.dot(x.T, (y_hat - y)) / num_train  # TODO 基于均方差损失对权重系数的一阶梯度
    db = np.sum(y_hat - y) / num_train  # 基于均方差损失对偏移量
    return y_hat, loss, dw, db


def linear_train(x, y, learning_rate=0.01, epochs=100, error = 0.00001):
    loss_his = []
    w, b = init_param(x.shape[1])

    for i in range(1, epochs):
        #     计算当前迭代的预测值，均方损失和梯度
        y_hat, loss, dw, db = linear_loss(x, y, w, b)
        w -= learning_rate * dw
        b -= learning_rate * db
        loss_his.append(loss)
        # print("epoch %d loss %.6f" % (i, loss))
        params = {"w": w, "b": b}
        grads = {"dw": dw, "db": db}

        if loss < error:
            # print("loss小于设定值，提前退出迭代")
            break
    return loss_his, params, grads


# 用w，b预测
def predict(x, params):
    w = params['w']
    b = params['b']
    y_pred = np.dot(w, x) + b
    return y_pred


# 画图

if __name__ == '__main__':
    pass

# %%
    x_train = np.array([[3,0],[3,1],[7,2]])
    y_train = np.array([[4],[5],[10]])

#%%
    loss_his, params, grads = linear_train(x_train,y_train,0.01,10000)
    print(params['w'])
    print(params['b'])
    print(min(loss_his))

    plt.figure(figsize=(8,6))
    plt.plot(loss_his)
    plt.xlabel('迭代次数')
    plt.ylabel('最小损失值')
    plt.title('最小损失值变化')
    plt.show()
#
# #%%
#     y_pred = predict(x_test, params)
#     print(y_pred)
#     print(y_test)
#     plt.figure()
#     plt.title('预测值比较')
#     plt.plot(y_pred,label = '预测值')
#     plt.plot(y_test,label = '实际值')
#     plt.legend()
#     plt.show()
#
# #%%
#     y_pred = y_pred.reshape(6,)
#     y_test = y_test.reshape(6,)
#     sns.lineplot(x = y_pred,y = y_test)
#
# #%%
#     print(r2_score(y_test,y_pred))
