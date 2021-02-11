import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import math

inf = 100000000000
start_label = 1


def vectorize(dataset):
    for feature in dataset:
        tmp_dict = {}
        for i in range(len(dataset[feature])):
            if dataset[feature][i] not in tmp_dict:
                tmp_dict[dataset[feature][i]] = len(tmp_dict)
        dataset[feature] = dataset[feature].map(tmp_dict)
    return dataset


def normalize(dataset):
    data_norm = (dataset - dataset.min()) / (dataset.max() - dataset.min())
    return data_norm


def euclidean(a, b):
    cur_sum = 0
    for i in range(len(a)):
        cur_sum += (a[i] - b[i]) ** 2
    return cur_sum ** 0.5


def manhattan(a, b):
    cur_sum = 0
    for i in range(len(a)):
        cur_sum += abs(a[i] - b[i])
    return cur_sum


def chebyshev(a, b):
    cur_max = 0
    for i in range(len(a)):
        cur_max = abs(a[i] - b[i]) if cur_max < abs(a[i] - b[i]) else cur_max
    return cur_max


def dbscan(trainX, eps, minPtsNum, distance):
    n = len(trainX)
    clusters = [-1 for _ in range(n)]
    visited = [False for _ in range(n)]
    curCluster = start_label - 1
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        curNeighbors = [(x, ind) for ind, x in enumerate(trainX) if eps > distance(x, trainX[i])]
        if len(curNeighbors) >= minPtsNum:
            curCluster += 1
            clusters[i] = curCluster
            for curX in curNeighbors:
                if not visited[curX[1]]:
                    visited[curX[1]] = True
                    curXNeighbors = [(x, ind) for ind, x in enumerate(trainX) if eps > distance(x, curX[0])]
                    if len(curXNeighbors) >= minPtsNum:
                        curNeighbors += curXNeighbors
                if clusters[curX[1]] == -1:
                    clusters[curX[1]] = curCluster

    return clusters, curCluster


def randIndex(curY, clusters):
    n = len(clusters)
    res = 0
    for i in range(n):
        for j in range(n):
            if (clusters[i] == clusters[j] and curY[i] == curY[j]) or (
                    clusters[i] != clusters[j] and curY[i] != curY[j]):
                res += 1
    return res / pow(n, 2)


def silhouetteIndex(curX, clusters, curCluster):
    def func_compactness(x_i, clust_ind):
        return sum([euclidean(x_i, x_j) for x_j in clustVec[clust_ind]]) / len(clustVec[clust_ind])

    def func_separability(x_i, clust_ind):
        return min(
            [func_compactness(x_i, curC) if curC != clust_ind else inf for curC in range(start_label, len(clustVec))])

    if curCluster == 1:
        return 0
    sil = 0
    n = 0
    clustVec = [[] for _ in range(0, curCluster + 1)]
    for (ind, cl) in enumerate(clusters):
        if cl != -1:
            n += 1
            curEl = curX[ind]
            clustVec[cl].append(curEl)
    for ind in range(start_label, len(clustVec)):
        for x in clustVec[ind]:
            a, b = func_compactness(x, ind), func_separability(x, ind)
            sil += (b - a) / max(a, b)

    return sil / (len(clusters))


def draw_plot(x, y, measure):
    fig, ax = plt.subplots()
    plt.plot(x, y)
    ax.set_xlabel("Radius")
    ax.set_ylabel(measure)


def draw_clusters(X, clusters, title, add=None):
    fig, ax = plt.subplots()
    colours = ["purple", "blue", "green", "brown", "pink", "olive", "cyan", "gray", "navy", "lime", "orange",
               "olivedrab"]
    labels = ["Класс#" + str(i) for i in range(1, len(clusters) + 1)]
    if add:
        labels = ["шумы"] + labels
        colours = ["r"] + colours
    for ind, cluster in enumerate(clusters):
        X_in_cluster = [X[i] for i in cluster]
        ax.scatter([x[0] for x in X_in_cluster], [x[1] for x in X_in_cluster], color=colours[ind], s=10,
                   label=labels[ind])

    fig.canvas.set_window_title(title)
    plt.legend()
    plt.title(title)


def main():
    dataset = pd.read_csv("wine.csv")
    class_name = "class"
    classes_num = 3
    curY = dataset[class_name].tolist()
    del dataset[class_name]
    dataset_normalized = normalize(dataset)
    curX = dataset_normalized.values.tolist()

    M = 10
    max_measure_out, max_measure_in = 0, 0
    best_radius_out, best_radius_in = 0, 0
    radii = [i / 100 for i in range(10, 100, 5)]
    rands, sils = [], []
    for radius in radii:
        clusters, curCluster = dbscan(curX, radius, M, euclidean)
        curRand = randIndex(curY, clusters)
        curSil = silhouetteIndex(curX, clusters, curCluster)
        rands.append(curRand)
        sils.append(curSil)
        print("Rand Index =", curRand, " for radius:" + str(radius))
        print("Silhouette Index =", curSil, " for radius:" + str(radius))
        if max_measure_out < curRand:
            max_measure_out = curRand
            best_radius_out = radius
        if max_measure_in < curSil:
            max_measure_in = curSil
            best_radius_in = radius

    print("Best Rand Index =", max_measure_out, " for radius:" + str(best_radius_out))
    print("Best Silhouette Index =", max_measure_in, " for radius:" + str(best_radius_in))
    draw_plot(radii, rands, "Rand")
    draw_plot(radii, sils, "Silhouette")

    clusters, curCluster = dbscan(curX, best_radius_out, M, euclidean)
    pca = PCA(n_components=2)
    transformed_X = pca.fit_transform(curX)

    labels = [i for i in range(start_label, classes_num + 1)]
    labels_cluster = [-1] + [i for i in range(start_label, curCluster + 1)]
    real_labels = [[i for i in range(len(curY)) if curY[i] == label] for label in labels]
    real_labels_cl = [[i for i in range(len(clusters)) if clusters[i] == label] for label in labels_cluster]

    draw_clusters(transformed_X, real_labels, 'Real data')
    draw_clusters(transformed_X, real_labels_cl, 'Clust', True)

    plt.show()


main()
