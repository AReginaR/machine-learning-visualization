import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from sklearn import datasets

layout = go.Layout(
    hovermode='closest',
    legend=dict(x=0, y=-0.01, orientation="h"),
    margin=dict(l=0, r=0, t=0, b=0),
    height=280,
)
#covmat = list()
color = {0: 'deepskyblue', 1: 'lightcoral', 2: 'limegreen', 3: 'gold', 4: 'plum', 5: 'lightpink',
         6: 'mediumpurple', 7: 'chocolate', 8: 'orange'}

def generate_data_pca(size, shape):
    if shape == 'blobs':
        return datasets.make_blobs(n_samples=size, n_features=3)

    elif shape == 'linear':
        return datasets.make_classification(n_samples=size, n_features=3, n_redundant=0, n_informative=2,
                                            random_state=2,
                                            n_clusters_per_class=1)

    elif shape == 'iris':
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        #    X_reduced = PCA(n_components=3).fit_transform(iris.data)
        return X, y

    elif shape == 'elips':
        x = np.arange(1, 51)
        y = 3 * x + np.random.randn(50) * 3
        z = 2 * x + np.random.randn(50) * 2
        X = []
        for i in range(20):
            X.append([x[i], y[i], z[i]])
        X = np.array(X)
        y = np.zeros(50)
        return X, y


def get_viz_data__(X, y):
    df = px.data.iris()
    features = ["sepal_width", "sepal_length", "petal_width", "petal_length"]

    fig = px.scatter_matrix(
        df,
        dimensions=features,
        color="species"
    )
    fig.update_traces(diagonal_visible=False)

    return fig


def cov_mat(covmat, values):
    for i in range(len(covmat)):
        covmat[i] = [round(v, 2) for v in covmat[i]]
    return go.Figure(data=[go.Table(header=dict(values=values),
                                    cells=dict(values=covmat))], layout=layout)


def get_viz_data(X, y, mean, v):
    trace = list()
    trace.append(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode='markers', name='Начальные данные',
                              marker=dict(color=[color[i] for i in y],
                                          size=10, symbol='circle', opacity=0.8, line=dict(width=3, color='white'))))
    for i in range(len(v)):
        trace.append(go.Scatter3d(
            x=[mean[0], (mean[0] + v[i][0])],
            y=[mean[1], (mean[1] + v[i][1])],
            z=[mean[2], (mean[2] + v[i][2])],
            mode='lines',
            line=dict(color='black', width=5),
            name=f"PC{i + 1}"
        ))
    return trace


def get_viz_res(X_new, y, loadings, text):
    trace2 = go.Scatter(
        x=X_new[:, 0],
        y=X_new[:, 1],
        mode='markers',
        marker=dict(color=[color[i] for i in y],
                    size=10, symbol='circle', opacity=0.8, line=dict(width=3, color='white')),
        name='данные',
    )
    fig = go.Figure(data=trace2, layout=layout)

    for i in range(len(loadings)):
        fig.add_shape(
            type='line', x0=0, y0=0,
            x1=loadings[i, 0], y1=loadings[i, 1], name='Старые оси',
        )
        fig.add_annotation(
            x=loadings[i, 0],
            y=loadings[i, 1],
            ax=0, ay=0,
            xanchor="center",
            yanchor="bottom",
            text=text[i],
        )
    return fig


def viz_bar(ex_var, ex_var_ratio):
    var, var2 = [], []
    for i in range(len(ex_var_ratio)):
        var.append(int(ex_var_ratio[i] * 100))
        if i > 0:
            var2.append(int(ex_var_ratio[i] * 100) + var[i - 1])
    pc = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', ]

    layout2 = go.Layout(
        hovermode='closest',
        yaxis=dict(title='Доля дисперсии'),
        legend=dict(x=0, y=-0.01, orientation="h"),
        margin=dict(l=0, r=0, t=0, b=0),
        height=250,
    )

    fig = go.Figure(
        data=[go.Bar(y=var,
                     x=[pc[i] for i in range(len(var))])],
        layout=layout2
    )
    return fig
