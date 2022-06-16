import json

import dash
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.Div('Алгоритм :', style={'marginLeft': 2}),
                    dcc.Dropdown(
                        id='algorithm',
                        options=[
                            {'label': 'K-means', 'value': 'k-means'},
                            {'label': 'SVM', 'value': 'svm'},
                            {'label': 'PCA', 'value': 'pca'},
                            {'label': 'Поиск аномалий', 'value': 'anomaly'}
                        ],
                        value='pca'
                    )
                ], style={'marginTop': 10, 'marginLeft': 10, 'marginRight': 10, 'marginBottom': 10}),

                html.Div([
                    html.Div([
                        html.H6('Генерация данных',
                                style={'text-align': 'center', 'marginTop': 10, 'marginBottom': 10, 'fontSize': 15,
                                       'fontWeight': 800}),

                        html.Div([
                            html.Div('Форма :', style={'marginLeft': 2}),
                            dcc.Dropdown(
                                id='dataset-choice',
                                options=[
                                    {'label': 'Круги', 'value': 'blobs'},
                                    {'label': 'Ирисы', 'value': 'iris'},
                                    {'label': 'Элипс', 'value': 'elips'},
                                    {'label': 'Рандом', 'value': 'linear'},
                                ],
                                value='blobs'
                            )
                        ], style={'marginLeft': 10, 'marginRight': 10, 'marginBottom': 10}),

                        html.Div([
                            html.Div('Размер :', style={'marginLeft': 2}),
                            dcc.Slider(id='size-slider', min=50, max=500, step=50, value=200,
                                       marks={50: '50', 500: '500'})
                        ], style={'marginLeft': 10, 'marginRight': 10, 'marginBottom': 10}),

                        html.Div([
                            dcc.Upload(id='upload-data', children=html.Div(['Или ', html.A('загрузите файлы')]),
                                       style={
                                           'width': '90%',
                                           'height': '40px',
                                           'lineHeight': '40px',
                                           'borderWidth': '1px',
                                           'borderStyle': 'dashed',
                                           'borderRadius': '5px',
                                           'textAlign': 'center',
                                           'margin': '10px'},
                                       # Allow multiple files to be uploaded
                                       multiple=True),
                        ], style={'marginLeft': 10, 'marginRight': 10}),

                    ]),

                    html.Div([
                        html.H6('Парамметры PCA',
                                style={'text-align': 'center', 'marginBottom': 10, 'fontSize': 15, 'fontWeight': 800}),

                        html.Div([
                            html.Div('Количество компонент:', style={'marginLeft': 2}),
                            dcc.Slider(id='n_component', min=2, max=3, value=2,
                                       marks={2: '2', 3: '3'})
                        ], style={'marginLeft': 10, 'marginRight': 10, 'marginBottom': 5}),

                        html.Div([
                            html.Div('svd_solver:', style={'marginLeft': 2}),
                            dcc.Dropdown(id='svd-solver',
                                         options=[
                                             {'label': 'auto', 'value': 'auto'},
                                             {'label': 'full', 'value': 'full'},
                                             {'label': 'arpack', 'value': 'arpack'},
                                             {'label': 'randomized', 'value': 'randomized'},
                                         ],
                                         value='auto')
                        ], style={'marginLeft': 10, 'marginRight': 10, 'marginBottom': 5})

                    ]),
                ], id='svm_form'),

            ], className='three columns',
                style={'background-color': 'white', 'height': '100%', 'borderStyle': 'solid', 'border-radius': 10,
                       'borderWidth': 2, 'borderColor': 'black'}),

            html.Div([

                html.H6('Данные до уменьшения размерности',
                        style={'text-align': 'center', 'marginTop': 15, 'marginBottom': 5, 'fontSize': 15,
                               'fontWeight': 800}),
                dcc.Graph(id='pca-graph', animate=True, config={'displayModeBar': False},
                          style={'marginLeft': 5, 'marginRight': 5, 'marginTop': 5}),
                html.H6('Результат алгоритма',
                        style={'text-align': 'center', 'marginTop': 15, 'marginBottom': 5, 'fontSize': 15,
                               'fontWeight': 800}),
                dcc.Graph(id='pca-graph-res', animate=True, config={'displayModeBar': False},
                          style={'marginLeft': 5, 'marginRight': 5, 'marginTop': 5}),

                dcc.Interval(id='interval-component', interval=3600 * 1000, n_intervals=0)

            ], className='nine columns',
                style={'background-color': 'white', 'height': '100%', 'border-width': 2, 'borderStyle': 'solid',
                       'border-radius': 10, 'borderColor': 'black'}),

        ], style={'height': '84%', 'position': 'relative', 'top': '1%'}),
    ], className='eight columns', style={'height': '100%'}),

    html.Div([
        html.Div([
            html.Div([
                html.H6('Процент выборочной диспресии',
                        style={'text-align': 'center', 'marginTop': 15, 'marginBottom': 5, 'fontSize': 15,
                               'fontWeight': 800}),
                dcc.Graph(id='viz-variance', config={'displayModeBar': False},
                          style={'marginLeft': 5, 'marginRight': 5}),
            ], style={'background-color': 'white', 'height': '49%', 'border-radius': 10, 'borderStyle': 'solid',
                      'borderWidth': 2, 'borderColor': 'black'}),

            html.Div([
                html.H6('Ковариациная матрица',
                        style={'text-align': 'center', 'marginTop': 15, 'marginBottom': 5, 'fontSize': 15,
                               'fontWeight': 800}),
                dcc.Graph(id='cov-mat', style={'marginLeft': 5, 'marginRight': 5}),
            ], style={'background-color': 'white', 'height': '49%', 'border-radius': 10, 'borderStyle': 'solid',
                      'borderWidth': 2, 'borderColor': 'black'}),

        ], style={'height': '84%', 'position': 'relative', 'top': '1%'}),
    ], className='four columns', style={'height': '100%', 'marginLeft': '1.5%', 'width': '32.5%', 'top': '1%'}),

    html.Div([
        html.Div(id='dataset-x', style={'display': 'none'}),
        html.Div(id='dataset-y', style={'display': 'none'}),
        html.Div(id='svm_counter', style={'display': 'none'}),
        html.Div(id='output-data-upload'),
    ], className='twelve columns', style={'marginTop': '1%'})

], style={'height': 820})


@app.callback([Output('dataset-x', 'children'), Output('dataset-y', 'children')],
              [Input('dataset-choice', 'value'), Input('size-slider', 'value')])
def update_dataset(shape, size):
    X, y = generate_data(size=size, shape=shape)
    X = json.dumps(X.tolist())
    y = json.dumps(y.tolist())
    return X, y


def generate_data(size, shape):
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


covmat = list()
color = {0: 'deepskyblue', 1: 'lightcoral', 2: 'limegreen', 3: 'gold', 4: 'plum', 5: 'lightpink',
         6: 'mediumpurple', 7: 'chocolate', 8: 'orange'}
layout = go.Layout(
    hovermode='closest',
    legend=dict(x=0, y=-0.01, orientation="h"),
    margin=dict(l=0, r=0, t=0, b=0),
    height=300,
)


@app.callback([Output('pca-graph', 'figure'),
               Output('pca-graph-res', 'figure'),
               Output('viz-variance', 'figure'),
               Output('cov-mat', 'figure')],
              [Input('dataset-x', 'children'),
               Input('dataset-y', 'children'),
               Input('n_component', 'value'),
               Input('svd-solver', 'value')
               ])
def update_pca(X, y, n_components, svd_solver):
    global covmat
    X = np.array(json.loads(X))
    y = np.array(json.loads(y))
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    text = ['x', 'y', 'z']

    if X.shape[1] == 3:
        pca = PCA(n_components=2, svd_solver=svd_solver)
        X_new = pca.fit_transform(X)

        covmat = pca.get_covariance()

        fig_cov = cov_mat(covmat, text)
        v = pca.components_
        mean = pca.mean_
        ex_var = pca.explained_variance_
        ex_var_ratio = pca.explained_variance_ratio_
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

        fig_var = viz_bar(ex_var, ex_var_ratio)

        trace = get_viz_data(X, y, mean, v)

        fig = get_viz_res(X_new, y, loadings, text)

        return go.Figure(data=trace, layout=layout), fig, fig_var, fig_cov

    elif X.shape[1] > 3:
        if n_components > X.shape[1]:
            return Exception()
        pca0 = PCA(svd_solver=svd_solver)
        xx = pca0.fit_transform(X)
        pca = PCA(n_components=n_components, svd_solver=svd_solver)
        X_new = pca.fit_transform(X)

        covmat = pca.get_covariance()
        features = ["sepal_width", "sepal_length", "petal_width", "petal_length"]
        fig_cov = cov_mat(covmat, features)

        v = pca.components_
        mean = pca.mean_
        ex_var = pca0.explained_variance_
        ex_var_ratio = pca0.explained_variance_ratio_
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

        fig_var = viz_bar(ex_var, ex_var_ratio)

        features = ["sepal_width", "sepal_length", "petal_width", "petal_length"]
        fig = get_viz_res(X_new, y, loadings, features)

        fig_1 = get_viz_data__(X, y)

        return fig_1, fig, fig_var, fig_cov


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
        covmat[i] = [round(v,2) for v in covmat[i]]
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


if __name__ == '__main__':
    app.run_server(debug=True)
