import numpy as np
import pandas as pd

from itertools import combinations, permutations
from operator import itemgetter

#%%
def trans_df2dict(df):
    """将数据转化成字典格式"""
    user_rating = dict()  # 用户评分数据
    for row in df.values:
        user_id, movie_id, rating = row[0], row[1], row[2]
        if user_id not in user_rating.keys():
            user_rating[user_id] = {}
        user_rating[user_id][movie_id] = rating
    return user_rating
#%%

def get_items_similarity(df, item_num):
    """计算items相似性矩阵，返回相似性矩阵"""
    # 1. 建立用户-物品的倒排表
    inverted_table = df.groupby(by='userid')['moveid'].agg(list).to_dict()

    # 2. 初始化共现矩阵，遍历每个用户，将物品两两组合，并在共现矩阵中加1
    W = np.zeros((item_num, item_num))

    # 统计每个电影被多少人看过
    count_item_users_num = df.groupby(by='moveid')['userid'].agg('count').to_dict()

    for key, val in inverted_table.items():
        val.sort(reverse=True)  # 降序
        com = combinations(val, 2)
        com_list = []
        for i in com:
            com_list.append(i)
        for per in combinations(val, 2):
            W[per[0] - 1][per[1] - 1] += 1
            W[per[1] - 1][per[0] - 1] += 1

    # 计算相似性
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i][j] /= np.sqrt(count_item_users_num.get(i + 1) * count_item_users_num.get(j + 1))

    w_dict = {}
    for i in range(W.shape[0]):
        tmp = []
        for index, k in enumerate(W[i]):
            tmp.append((index + 1, k))
        w_dict[i + 1] = tmp
    return w_dict
#%%
df = pd.read_csv(r'D:\PythonEx\machinelearningIntro\机器学习实践\关联规则\data\ratings.csv').head(100)
#%%
user_rating = trans_df2dict(df)
#%%
inverted_table = df.groupby(by='userid')['moveid'].agg(list).to_dict()
#%%
item_num = df.moveid.nunique()
#%%
w_dict = get_items_similarity(df, item_num)
w_dict
