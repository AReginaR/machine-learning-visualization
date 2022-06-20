import numpy as np
import plotly.graph_objs as go
from sklearn import datasets

color = {-1: 'deepskyblue', 1: 'lightcoral'}
color_label = [[0, '#FF0000'], [1, '#0000FF']]


def viz_anomaly(X, y, xx, yy, Z):
    trace1 = go.Scatter(
        x=X[:, 0], y=X[:, 1], mode='markers', name=f'Данные',
        marker=dict(size=10, symbol='circle', color=[color[i] for i in y],
                    colorscale=color_label, line=dict(width=1))
    )
    trace2 = go.Contour(
        x=np.arange(xx.min(), xx.max()), y=np.arange(yy.min(), yy.max()),
        z=Z.reshape(xx.shape), showscale=False, hoverinfo='none',
        contours=dict(showlines=False, type='constraint', operation='=', ),
        name=f'Гиперплоскость', line=dict(color='black')
    )
    return [trace1, trace2]


def viz_anomaly2(X, xx, yy, Z):
    trace1 = go.Scatter(
        x=X[:, 0], y=X[:, 1], mode='markers', name=f'Данные',
        marker=dict(size=10, symbol='circle',
                    colorscale=color_label, line=dict(width=1))
    )
    trace2 = go.Contour(
        x=np.arange(xx.min(), xx.max()), y=np.arange(yy.min(), yy.max()),
        z=Z.reshape(xx.shape), showscale=False, hoverinfo='none',
        contours=dict(showlines=False, type='constraint', operation='=', ),
        name=f'Гиперплоскость', line=dict(color='black')
    )
    return [trace1, trace2]


def generate_data_anomaly(size, shape, noise):
    if shape == 'moon':
        return datasets.make_moons(n_samples=size, noise=noise, random_state=0)

    elif shape == 'circle':
        return datasets.make_circles(n_samples=size, noise=noise, factor=0.5, random_state=1)

    elif shape == 'blobs':
        return datasets.make_blobs(centers=[[2, 2], [-2, 2]], cluster_std=0.5, n_samples=size)

    elif shape == 'linear':
        X, y = datasets.make_classification(n_samples=size, n_features=2, n_redundant=0, n_informative=2,
                                            random_state=2,
                                            n_clusters_per_class=1)

        rng = np.random.RandomState(2)
        X += noise * rng.uniform(size=X.shape)
        linearly_separable = (X, y)

        return linearly_separable
