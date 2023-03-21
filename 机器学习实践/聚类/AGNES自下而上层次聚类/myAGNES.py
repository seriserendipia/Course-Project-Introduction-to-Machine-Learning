import pandas as pd
# 欧氏距离
import numpy as np

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
    return [[i] for i in x]


def init_distance_matrix(clusters, dist_between_two_clusters_method = avgdist):
    distance_matrix = np.full((len(clusters),len(clusters)), np.inf)
    for cluster1_index in range(len(clusters)):
        for cluster2_index in range(cluster1_index + 1, len(clusters)):
            dist_between_two_clusters = dist_between_two_clusters_method(clusters[cluster1_index],clusters[cluster2_index])
            distance_matrix[cluster1_index][cluster2_index] = dist_between_two_clusters
    return distance_matrix

def update_distance_matrix(x_dim,clusters,distance_matrix,cluster1_index,clusters2_index, dist_between_two_clusters_method = avgdist):
    for i in range(x_dim):
        distance_matrix = np.delete(distance_matrix,clusters2_index,axis= i)

    for other_cluster_index in range(len(clusters)):
        dist_between_two_clusters = dist_between_two_clusters_method(clusters[cluster1_index],clusters[other_cluster_index])
        if other_cluster_index > cluster1_index:
            distance_matrix[cluster1_index][other_cluster_index] = dist_between_two_clusters
        elif other_cluster_index < cluster1_index:
            distance_matrix[other_cluster_index][cluster1_index] = dist_between_two_clusters
    return distance_matrix



def get_closest_two_clusters(distance_matrix):
    closest_two_clusters_index = np.unravel_index(np.argmin(distance_matrix, axis=None), distance_matrix.shape)
    return closest_two_clusters_index


class MyAGNESmodel:
    model:list
    distance_matrix:np.ndarray

    def __init__(self,x):
        self.clusters = init_clusters(x)
        self.distance_matrix = init_distance_matrix(self.clusters)

    def get_result_df(self):
        df = pd.DataFrame()
        for i in range(len(self.clusters)):
            dfi = pd.DataFrame(self.clusters[i])
            dfi["label"] = i

            df = pd.concat([df,dfi])

        return df


def new_clusters(clusters, cluster1_index, clusters2_index):
    clusters[cluster1_index].extend(clusters[clusters2_index])
    clusters.pop(clusters2_index)
    return clusters


def my_AGNES(x,terminal_k = 2):
    x = np.array(x)
    x_dim = x.shape[1]
    model = MyAGNESmodel(x)
    while len(model.clusters) > terminal_k:
        cluster1_index,clusters2_index = get_closest_two_clusters(model.distance_matrix)
        model.clusters = new_clusters(model.clusters,cluster1_index,clusters2_index)
        model.distance_matrix = update_distance_matrix(x_dim,model.clusters,model.distance_matrix,cluster1_index,clusters2_index)
    return model


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

    model = my_AGNES(train_data, terminal_k = 2)
    df = model.get_result_df()
    print(df)
