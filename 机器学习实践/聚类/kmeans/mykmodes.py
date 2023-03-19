from mykmeans import *

def hamming_distance(p1,p2):
    pass

# 计算新的簇中心,用簇内属性的众数
def get_new_centroids_by_mode(clusters, k, x, x_dimension):
    pass

def my_k_modes(x, k, max_iter_times=10, distance_method=hamming_distance):
    return my_k_means(x, k, max_iter_times=10,
                      distance_method=distance_method,
                      generate_new_controid_method=get_new_centroids_by_mode)
