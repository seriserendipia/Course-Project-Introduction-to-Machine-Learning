

import numpy as np


def standerlize(x):
    '''标准化（最大最小）'''
    for i in range(len(x)):
        max_min_diff = np.max(x[i]) - np.min(x[i])
        x[i] = np.divide((x[i] - np.min(x[i])),max_min_diff)
        print(x[i])
    print(x)
    return x

# 初始化,w


# 预测值，损失，更新的梯度


# 迭代次数
def my_linear_regression(x,max_iter = 100):
    x = np.array(x)
    x = standerlize(x)


# r2-score


# 用w，b预测

# 画图

if __name__ == '__main__':
    x = np.array([np.arange(10)]*3)
    print(standerlize(x))
