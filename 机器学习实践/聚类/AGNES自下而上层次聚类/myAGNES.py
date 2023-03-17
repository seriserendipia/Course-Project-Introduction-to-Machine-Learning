
# 欧氏距离
import numpy as np
import pandas as pd


def get_Euclidean_Distance(p1, p2):
    distance = np.sqrt(sum(np.square(p1 - p2)))
    return distance

def dist_between_two_points(i, j, dist_between_two_points_method = get_Euclidean_Distance):
    return dist_between_two_points_method(i,j)

def dist_between_clusters(cluster1, cluster2):
    return [dist_between_two_points(i,j) for i in cluster1 for j in cluster2]

def mindist(cluster1,cluster2):
    return min(dist_between_clusters(cluster1,cluster2))

def maxdist(cluster1,cluster2):
    return max(dist_between_clusters(cluster1,cluster2))

def avgdist(cluster1, cluster2):
    return sum(dist_between_clusters(cluster1,cluster2))/len(cluster1)*len(cluster2)


def init_clusters(x):
    return [i for i in x]


def init_distance_matrix(clusters, dist_between_two_clusters_method = avgdist):
    distance_matrix = np.zeros((len(clusters),len(clusters)))
    for cluster1_index in range(len(clusters)):
        for cluster2_index in range(cluster1_index + 1,len(clusters)):
            dist_between_two_clusters = dist_between_two_clusters_method(clusters[cluster1_index],clusters[cluster2_index])
            distance_matrix[cluster1_index][cluster2_index] = dist_between_two_clusters
    return distance_matrix

def update_distance_matrix(clusters,cluster1_index,clusters2_index):
    pass


def get_closest_two_clusters(clusters,distance_matrix):
    pass
    #TODOㅁㅁ return cluster1_index,clusters2_index


class MyAGNESmodel:

    def __init__(self,x):
        self.clusters = init_clusters(x)
        self.distance_matrix = init_distance_matrix(self.clusters)

    def show(self):
        df = pd.DataFrame()
        for i in range(len(self.clusters)):
            dfi = pd.DataFrame(self.clusters[i])
            dfi["label"] = i

            df = pd.concat([df,dfi])

        print(df)


def new_clusters(clusters, cluster1_index, clusters2_index):
    pass


def my_AGNES(x,terminal_k = 2):
    model = MyAGNESmodel(x)
    while len(model.clusters) > terminal_k:
        cluster1_index,clusters2_index = get_closest_two_clusters(model.clusters)
        model.clusters = new_clusters(model.clusters,cluster1_index,clusters2_index)
        model.distance_matrix = update_distance_matrix(model.clusters,cluster1_index,clusters2_index)
    model.show()
