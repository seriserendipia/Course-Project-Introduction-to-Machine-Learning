# 和老师的区别
# 没有单独的b项，在x上添加一列1
#

import numpy as np

def r2_score(y_test,y_pred):
    y_avg = np.mean(y_test)
    ss_tot = np.sum((y_test - y_avg)**2)
    ss_res = np.sum((y_test - y_pred)**2)
    r2 = 1 - (ss_res/ss_tot)
    return r2

def standerlize(x):
    '''标准化（最大最小）'''
    for i in range(len(x)):
        max_min_diff = np.max(x[i]) - np.min(x[i])
        x[i] = np.divide((x[i] - np.min(x[i])),max_min_diff)
        print(x[i])
    print(x)
    return x

# 初始化,w
def init_param(x_dim):
    w = np.ones((1,x_dim + 1))
    return w

# 预测值，损失，更新的梯度
def linear_regress():
    pass
    # return prediction, loss, gredient


# 迭代次数
def my_linear_regression(x,max_iter = 100):
    x = np.array(x)
    x = standerlize(x)




# 用w，b预测
def predict(x,w):
    y = np.dot(w,x)
    return y

# 画图

if __name__ == '__main__':
    x = np.array([np.arange(10)]*3)
    print(standerlize(x))
