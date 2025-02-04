import random

# 随机浮点数，范围[0,1)
import numpy as np

random.random()

# 随机浮点数，范围[a,b)
random.uniform(a,b)

# 随机整数
random.randint(a,b)

# 随机从指定范围步长中获得一个整数
random.randrange(start=,stop=,step=)

# 从指定序列中随机挑选一个元素
random.choice([seq])

# 打乱，随机排序seq
random.shuffle([seq])

# 从指定序列中随机获取指定长度的片段
random.sample(sequence,k)

# 返回n*m*l*……的高维随机数组矩阵，取值[0,1)
np.random.rand(第一维的长度，第二维的长度，……)

# 返回高斯分布，平均值，方差，
np.random.normal(loc=mean,scale=std,size=)
