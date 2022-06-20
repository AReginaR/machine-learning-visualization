import numpy as np
import plotly.graph_objs as go
from sklearn import datasets
from sklearn import metrics


def generate_data_svm(size, shape, noise):
    if shape == 'moon':
        return datasets.make_moons(n_samples=size, noise=noise, random_state=0)

    elif shape == 'circle':
        return datasets.make_circles(n_samples=size, noise=noise, factor=0.5, random_state=1)

    elif shape == 'linear':
        X, y = datasets.make_classification(n_samples=size, n_features=2, n_redundant=0, n_informative=3,
                                            random_state=2,
                                            n_clusters_per_class=2)

        rng = np.random.RandomState(2)
        X += noise * rng.uniform(size=X.shape)
        linearly_separable = (X, y)

        return linearly_separable


def viz_svm(X_train, X_test, y_train, y_test, Z, xx, yy, mesh_step):
    color_label = [[0, '#FF0000'], [1, '#0000FF']]
    color = {0: 'deepskyblue', 1: 'lightcoral', 2: 'limegreen', 3: 'gold', 4: 'plum', 5: 'lightpink',
                     6: 'mediumpurple', 7: 'chocolate', 8: 'orange'}

    trace_line = go.Contour(
        x=np.arange(xx.min(), xx.max(), mesh_step), y=np.arange(yy.min(), yy.max(), mesh_step),
        z=Z.reshape(xx.shape), showscale=False, hoverinfo='none',
        contours=dict(showlines=False, type='constraint', operation='=', ),
        name=f'Гиперплоскость', line=dict(color='black')
    )

    trace_train = go.Scatter(
        x=X_train[:, 0], y=X_train[:, 1], mode='markers', name=f'Обучающие данные',
        marker=dict(size=10, symbol='circle', color=[color[i] for i in y_train],
                    colorscale=color_label, line=dict(width=1))
    )

    trace_test = go.Scatter(
        x=X_test[:, 0], y=X_test[:, 1], mode='markers', name=f'Тестовые данные',
        marker=dict(size=10, symbol='triangle-up', color=[color[i] for i in y_test],
                    colorscale=color_label, line=dict(width=1))
    )

    layout = go.Layout(
        xaxis=dict(ticks='', showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(ticks='', showticklabels=False, showgrid=False, zeroline=False),
        hovermode='closest',
        legend=dict(x=0, y=-0.01, orientation="h"),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    full = [trace_test, trace_train, trace_line]

    figure = go.Figure(data=full, layout=layout)

    return figure


def viz_roc_auc(clf, X_test, y_test):
    decision_test = clf.decision_function(X_test)
    fpr, tpr, threshold = metrics.roc_curve(y_test, decision_test)

    roc_auc = metrics.roc_auc_score(y_true=y_test, y_score=decision_test)

    trace = go.Scatter(x=fpr, y=tpr, mode='lines', name='Тестовые данные')

    layout = go.Layout(
        title=f'ROC Кривая (AUC = {roc_auc:.3f})',
        xaxis=dict(title='False Positive Rate'),
        yaxis=dict(title='True Positive Rate'),
        legend=dict(x=0, y=1.05, orientation="h"),
        margin=dict(l=50, r=10, t=55, b=40),
        height=250,
    )

    figure = go.Figure(data=trace, layout=layout)

    return figure
