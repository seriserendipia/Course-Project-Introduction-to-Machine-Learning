import numpy as np

class Point():

    def __init__(self,x):
        self.visited = 0
        self.neighbor = []
        self.attr = x
        self.belong_to_cluster_no = -1

    def __str__(self):
        return str(self.attr)

def get_Euclidean_Distance(p1, p2):
    distance = np.sqrt(sum(np.square(p1 - p2)))
    return distance


def is_core_point(point,min_samples):
    return len(point.neighbor) >= min_samples


def my_DBSCAN(x,eps = 0.005, min_samples = 5):
    x = np.array(x)
    point_list = [Point(x[i]) for i in range(len(x))]
    core_object_list = []

    # 创建每个样本点的领域点index列表
    for i in range(len(x)):
        for j in range(len(x)):
            dist = get_Euclidean_Distance(x[i],x[j])
            if dist <= eps:
                point_list[i].neighbor.append(point_list[j])

    # 创建核心点列表
    for i in range(len(x)):
        if is_core_point(point_list[i],min_samples = min_samples):
            core_object_list.append(point_list[i])

    # 聚类
    clusters_no = 0
    while(len(core_object_list) != 0):
        core_point = core_object_list.pop()
        if core_point.visited == 0:
            ready_to_be_checked_list = core_point.neighbor
            for i in ready_to_be_checked_list:
                i.visited = 1

            while (len(ready_to_be_checked_list) != 0):
                point = ready_to_be_checked_list.pop()
                point.belong_to_cluster_no = clusters_no
                point.visited = 1
                if is_core_point(point,min_samples=min_samples):
                    for i in point.neighbor:
                        if i.visited == 0:
                            ready_to_be_checked_list.append(i)

            clusters_no = clusters_no + 1

    print(f"聚类的数量为: {clusters_no}")
    return [point.belong_to_cluster_no for point in point_list]


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

    clusters = my_DBSCAN(train_data,eps=1,min_samples=2)
    print(f"聚类的数量{clusters}")
