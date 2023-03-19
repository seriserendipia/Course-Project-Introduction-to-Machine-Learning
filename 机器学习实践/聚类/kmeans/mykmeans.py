import numpy as np
import pandas as pd

np.random.seed(123456)

class MyKmeansModel:
    clusters = None

    # 随机初始化中心点
    def init_centroids(self,x, k, x_dimension):
        random_centroids_index = np.random.choice(range(len(x)), k, False)
        centroids = np.zeros((k, x_dimension))
        for i in range(k):
            centroids[i] = x[random_centroids_index[i]]
        return centroids

    def __init__(self,x,k,x_dimension):
        self.centroids = self.init_centroids(x,k,x_dimension)

    def get_result_df(self):
        df = pd.DataFrame()
        i = 0
        for centroid,points in zip(self.centroids,self.clusters):
            # print(f"第{i}类的中心点{centroid}")

            dfi = pd.DataFrame(points)
            dfi["label"] = i

            df = pd.concat([df,dfi])
            i = i + 1

        return df


# 欧氏距离
def get_Euclidean_Distance(p1, p2):
    distance = np.sqrt(sum(np.square(p1 - p2)))
    return distance


# 计算两点距离
def distance_between_points(p1,p2,distance_method):
    distance = distance_method(p1,p2)
    return distance

# 计算新的簇中心
def generate_new_centroids(clusters,k,x,x_dimension):
    new_centroids = np.zeros((k, x_dimension))
    for i in range(len(clusters)):
        cluster_arr = np.array(clusters[i])
        new_centroid = np.average(cluster_arr, axis=0)
        new_centroids[i] = new_centroid
    return new_centroids

# 两个点的差的平方和
def square_diff(p1, p2):
    # print(p1)
    # print(p2)
    a = sum(np.square(p1 - p2))
    # print(f"两个点的差的平方和{a}")
    return a

# 计算SSE
def calculate_SSE(clusters,centroids):
    sse = 0
    for centroid,cluster in zip(centroids,clusters):
        sse_in_one_cluster = 0
        # print(centroid,cluster)
        sse_in_one_cluster += sum([square_diff(i,centroid) for i in cluster])
        # sse_in_one_cluster = sse_in_one_cluster/(len(cluster))
        sse_in_one_cluster = sse_in_one_cluster
        sse += sse_in_one_cluster
    # print(sse)
    return sse

# 建簇
def bulid_clusters(x, k,centroids,distance_method):
    clusters = [[] for i in range(k)]
    # print(clusters)
    for i in x:
        centroids_i_distance_arr = [distance_between_points(i, centroid,distance_method) for centroid in centroids]
        # print("一个点到各个中心的距离列表")
        # print(centroids_i_distance_arr)
        centroid_index =  np.argmin(centroids_i_distance_arr)
        # print(f"这个点所属的中心{centroid_index}")
        clusters[centroid_index].append(i)
    return clusters


def my_k_means(x:np.ndarray,k,max_iter_times = 10,distance_method = get_Euclidean_Distance):
    x = np.array(x)
    x_dimension = x.shape[1]
    model = MyKmeansModel(x,k,x_dimension)
    sse = 0
    for i in range(max_iter_times):
        model.clusters = bulid_clusters(x,k,model.centroids,distance_method)
        model.centroids = generate_new_centroids(model.clusters,k,x,x_dimension)
        model.sse = calculate_SSE(model.clusters,model.centroids)
        # model.get_result_df()
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

    model = my_k_means(train_data, k=3)
    df = model.get_result_df()
    print(f"df")
