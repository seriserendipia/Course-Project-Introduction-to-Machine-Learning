import numpy as np

from mykmeans import *


def get_new_centroids_by_median(clusters, k, x, x_dimension):
    """
    计算新的簇中心,用簇内点中最接近中心的点
    :param clusters:
    :param k:
    :param x:
    :param x_dimension:
    :return:
    """
    new_centroids = np.zeros((k, x_dimension))
    for i in range(k):
        cluster = clusters[i]
        sum_distance_between_one_point_and_others_list = [get_distance_with_others(point,cluster,distance_method=get_Euclidean_Distance) for point in cluster]
        new_centroid_index = np.argmin(sum_distance_between_one_point_and_others_list)
        new_centroid = cluster[new_centroid_index]
        new_centroids[i] = new_centroid

    return new_centroids

def my_k_medoids(x,k,max_iter_times=10):
    return my_k_means(x, k,
                      max_iter_times=10,
                      generate_new_controid_method=get_new_centroids_by_median)

if __name__ == '__main__':
    import pandas as pd

    # train_data = pd.DataFrame([
    #     [0, 0],
    #     [1, 0],
    #     [2, 0],
    #     [0, 1],
    #     [3, 1],
    #     [2, 2],
    #     [3, 2],
    # ])

    datasavedir = r"D:\PythonEx\machinelearningIntro\data\SP500Data-月收益率平均和方差.xlsx"
    data7 = pd.read_excel(datasavedir)
    data7.index = data7.iloc[:, 0]
    data7 = data7.drop("Unnamed: 0", axis=1)
    train_data = data7

    model = my_k_medoids(train_data, k=3)
    df = model.get_result_df()
    print(f"{df}")
