import numpy as np
import plotly.graph_objs as go
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

color_cluster = {0: 'deepskyblue', 1: 'lightcoral', 2: 'limegreen', 3: 'gold', 4: 'plum', 5: 'lightpink',
                 6: 'mediumpurple', 7: 'chocolate', 8: 'orange'}


def k_means_score(data):
    sse = list()
    silhouette = list()
    silhouette_traces = list()
    for n in range(2, 10):
        kmeans = KMeans(n_clusters=n, random_state=0)
        kmeans.fit(data)
        preds = kmeans.predict(data)

        sse.append(kmeans.inertia_)

        silhouette.append(silhouette_score(data, preds))

        sample_silhouette_values = silhouette_samples(data, preds)
        y_lower = 10
        silhouette_traces_i = list()

        for i in range(n):
            ith_cluster_silhouette_values = sample_silhouette_values[preds == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            filled_area = go.Scatter(y=np.arange(y_lower, y_upper),
                                     x=ith_cluster_silhouette_values,
                                     mode='lines',
                                     showlegend=False,
                                     line=dict(width=1, color=color_cluster[i]),
                                     fill='tozerox',
                                     fillcolor=color_cluster[i])
            silhouette_traces_i.append(filled_area)
            y_lower = y_upper + 10

        trace_line = go.Scatter(
            x=[silhouette[n - 2], silhouette[n - 2]],
            y=[0, 350],
            mode='lines',
            text='Среднее',
            line=dict(width=1, color='black', ),

        )

        silhouette_traces_i.append(trace_line)
        silhouette_traces.append(silhouette_traces_i)
    return sse, silhouette, silhouette_traces


def generate_data(shape='gmm', size=200, n_clusters=3):
    cluster_std = [0.5, 1, 1.3] * 3
    cluster_std = cluster_std[:n_clusters]

    if shape == 'gmm':
        data, y = datasets.make_blobs(n_samples=size, n_features=2, centers=n_clusters, cluster_std=cluster_std)

    elif shape == 'circle':
        data, y = datasets.make_circles(n_samples=size, factor=0.5, noise=0.05)

    elif shape == 'moon':
        data, y = datasets.make_moons(n_samples=size, noise=0.1)

    elif shape == 'anisotropic':
        transformations = {0: [[0.6, -0.6], [-0.4, 0.8]], 1: [[-0.7, -0.6], [0.6, 0.8]], 2: [[0.8, -0.1], [0.8, 0.1]]}
        data, y = datasets.make_blobs(n_samples=size, n_features=2, centers=n_clusters, cluster_std=cluster_std)
        for i in range(n_clusters):
            data[y == i] = np.dot(data[y == i], transformations[i % 3])
        data = 5 * data

    else:
        data = 30 * np.random.rand(size, 2) - 15

    data[:, 0] = 30 * (data[:, 0] - min(data[:, 0])) / (max(data[:, 0]) - min(data[:, 0])) - 15
    data[:, 1] = 30 * (data[:, 1] - min(data[:, 1])) / (max(data[:, 1]) - min(data[:, 1])) - 15

    return data


def generate_centroids(data, k=3, init_method='random'):
    if init_method == 'random':
        centroids = 30 * np.random.rand(k, 2) - 15
    else:
        indices = list(range(data.shape[0]))
        centroids = np.empty((k, 2))
        centroids[0, :] = data[np.random.choice(indices), :]
        D = np.power(np.linalg.norm(data - centroids[0], axis=1), 2)
        P = D / D.sum()

        for i in range(k - 1):
            centroid_index = np.random.choice(indices, size=1, p=P)
            centroids[i + 1, :] = data[centroid_index, :]
            Dtemp = np.power(np.linalg.norm(data - centroids[-1, :], axis=1), 2)
            D = np.min(np.vstack((Dtemp, D)), axis=0)
            P = D / D.sum()

    return centroids


def expectation(data, centroids):
    k = centroids.shape[0]

    dist = np.empty((data.shape[0], k))

    for i in range(k):
        dist[:, i] = np.linalg.norm(data - centroids[i, :], axis=1)

    labels = dist.argmin(axis=1)
    return labels


def maximization(data, centroids, labels):
    k = centroids.shape[0]
    new = centroids

    for i in range(k):
        if sum(labels == i) > 0:
            new[i, :] = data[labels == i, :].mean(axis=0)

    return new


def viz(dataset, labels, centroids):
    if sum(labels) == -dataset.shape[0]:

        trace1 = go.Scatter(x=dataset[:, 0], y=dataset[:, 1], mode='markers', name='Датасет',
                            marker=dict(color='white', size=10, line=dict(width=1, color='gray')), )

        centroid_points = dataset.copy()
        for idx, i in enumerate(range(1, 3 * centroid_points.shape[0], 3)):
            centroid_points = np.insert(centroid_points, i, dataset.mean(axis=0), axis=0)
            centroid_points = np.insert(centroid_points, i + 1, np.array([None, None]), axis=0)

        trace2 = go.Scatter(
            x=centroid_points[:, 0],
            y=centroid_points[:, 1],
            name='Принадлежность',
            mode='lines',
            line=dict(color='darkgray', width=0.5)
        )

        trace3 = go.Scatter(
            x=centroids[:, 0],
            y=centroids[:, 1],
            name='Центры',
            mode='markers',
            marker=dict(color='white',
                        size=15, symbol='circle', opacity=0.8, line=dict(width=3, color='black'))
        )

        return [trace1, trace2, trace3]

    trace1 = go.Scatter(
        x=dataset[:, 0],
        y=dataset[:, 1],
        mode='markers',
        marker=dict(color=[color_cluster[i] for i in labels], size=10,
                    line=dict(width=1, color='white')),
        name='Датасет'
    )

    centroid_points = dataset.copy()
    for idx, i in enumerate(range(1, 3 * centroid_points.shape[0], 3)):
        centroid_points = np.insert(centroid_points, i, centroids[labels[idx], :], axis=0)
        centroid_points = np.insert(centroid_points, i + 1, np.array([None, None]), axis=0)

    trace2 = go.Scatter(
        x=centroid_points[:, 0],
        y=centroid_points[:, 1],
        mode='lines',
        line=dict(color='darkgray', width=.5),
        name='Принадлежность'
    )

    trace3 = go.Scatter(
        x=centroids[:, 0],
        y=centroids[:, 1],
        name='Центры',
        mode='markers',
        marker=dict(color=list(color_cluster.values()),
                    size=15, symbol='circle', opacity=.8, line=dict(width=3, color='black'))
    )

    return [trace1, trace2, trace3]
