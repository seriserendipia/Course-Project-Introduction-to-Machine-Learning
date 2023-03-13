#!/usr/bin/env python
# coding: utf-8

# # 本地路径（自定义，按你的情况改）

#你放.ipynb文件和tool_func.py的文件夹路径
currentPath = "D:\PythonEx\machinelearningIntro"

# 单个的数据集路径
my_data_set_dir = "D:\大学资料\机器学习与数据挖掘\模型与数据\时间序列数据\证券数据\股票\收盘价.csv"

# 添加项目路径至path
import os
import sys
#currentPath = os.path.join(os.getcwd(),"machinelearningIntro","")
sys.path.append(currentPath)

#工具类
from tool_func import *


# # 小组作业要求
# 
#  选择20-30个回归数据集（王双成教授发给你们的数据集），在表中填入不同回归模型的回归误差 ， 回归误差保留4位小数。
# 
# 回归模型包括：1）计量经济模型：ARAMA、GARCH和VAR；2）循环神经网络：RNN、LSTM和GRU。
# 
# 对实验结果进行比较与分析
# 

# ## 导入数据集

# In[78]:


import pandas as pd
raw_data = pd.read_csv(my_data_set_dir)
raw_data


# ## 划分一个时间序列出来，去掉Nan和零值
data = raw_data.iloc[:,3]
data.index = raw_data["日期"]
data.index = pd.DatetimeIndex(data.index).to_period('D')

data = data.dropna()
data = data.drop( index = data[data == 0].index)
# data.value_counts()
#这个数据集时间顺序倒了，给它正过来
data = data.iloc[::-1]

# ## 划分训练集和测试集
#训练集是时间序列的前70%，测试集后30%
mask = int(len(data)*0.7)
train_data = data[0:mask].copy()
test_data = data[mask:-1].copy()



#%%
# ### LSTM


import math
import os

import numpy as np
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow import optimizers
