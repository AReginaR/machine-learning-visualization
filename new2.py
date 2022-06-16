import json

import dash
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output
from sklearn import datasets
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pyod.models.copod import COPOD

app = dash.Dash(__name__)
color = {-1: 'deepskyblue', 1: 'lightcoral'}
color_label = [[0, '#FF0000'], [1, '#0000FF']]
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
                        value='anomaly'
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
                                    {'label': 'Линейно', 'value': 'linear'},
                                    {'label': 'Концентрические круги', 'value': 'circle'},
                                    {'label': 'Луны', 'value': 'moon'},
                                    {'label': 'Blobs', 'value': 'blobs'},
                                ],
                                value='circle'
                            )
                        ], style={'marginLeft': 10, 'marginRight': 10, 'marginBottom': 10}),

                        html.Div([
                            html.Div('Размер :', style={'marginLeft': 2}),
                            dcc.Slider(id='size-slider', min=50, max=500, step=50, value=200,
                                       marks={50: '50', 500: '500'})
                        ], style={'marginLeft': 10, 'marginRight': 10, 'marginBottom': 10}),

                        html.Div([
                            html.Div('Шум :', style={'marginLeft': 2}),
                            dcc.Slider(id='noise-slider', min=0.0, max=1.0, step=0.1,
                                       marks={i / 10: str(i / 10) for i in range(0, 11, 2)},
                                       value=.1)
                        ], style={'marginLeft': 10, 'marginRight': 10}),

                    ]),

                    html.Div([
                        html.H6('Парамметры ',
                                style={'text-align': 'center', 'marginBottom': 10, 'fontSize': 15, 'fontWeight': 800}),

                        html.Div([
                            html.Div('Kernel:', style={'marginLeft': 2}),
                            dcc.Dropdown(
                                id='init',
                                options=[
                                    {'label': 'Linear', 'value': 'linear'},
                                    {'label': 'Polynomial', 'value': 'poly'},
                                    {'label': 'Radial basis function (RBF)', 'value': 'rbf'},
                                    {'label': 'Sigmoid', 'value': 'sigmoid'},
                                ],
                                value='rbf'
                            )
                        ], style={'marginLeft': 10, 'marginRight': 10, 'marginBottom': 10}),

                        html.Div([
                            html.Div('contamination :', style={'marginLeft': 2}),
                            dcc.Slider(id='contamination', min=0.1, max=1, value=0.2, step=0.1,
                                       marks={0.2: '0.2', 0.4: '0.4', 0.6: '0.6'})
                        ], style={'marginLeft': 10, 'marginRight': 10, 'marginBottom': 5}),

                        html.Div([
                            html.Div('Degree(Для Poly) :', style={'marginLeft': 2}),
                            dcc.Slider(id='degree', min=0, max=10, step=1, value=3)
                        ], style={'marginLeft': 10, 'marginRight': 10, 'marginBottom': 5}),

                        html.Div([
                            html.Div('gamma :', style={'marginLeft': 2}),
                            dcc.Slider(id='gamma', min=0.1, max=100, value=1,
                                       marks={0.1: '0.1', 1: '1', 10: '10', 100: '100'})
                        ], style={'marginLeft': 10, 'marginRight': 10, 'marginBottom': 5}),

                    ]),
                ], id='svm_form'),

            ], className='three columns',
                style={'background-color': 'white', 'height': '100%', 'borderStyle': 'solid', 'border-radius': 10,
                       'borderWidth': 2, 'borderColor': 'black'}),

            html.Div([
                html.H6('Isolation Forest',
                        style={'text-align': 'center', 'marginTop': 10, 'marginBottom': 5, 'fontSize': 15,
                               'fontWeight': 800, 'color': '#4B4D55'}),

                dcc.Graph(id='graph1', animate=True, config={'displayModeBar': False},
                          style={'marginLeft': 5, 'marginRight': 5, 'marginTop': 5}),

                html.H6('One-Class SVM',
                        style={'text-align': 'center', 'marginTop': 10, 'marginBottom': 5, 'fontSize': 15,
                               'fontWeight': 800, 'color': '#4B4D55'}),

                dcc.Graph(id='graph2', animate=True, config={'displayModeBar': False},
                          style={'marginLeft': 5, 'marginRight': 5, 'marginTop': 5}),

                html.H6('Copod',
                        style={'text-align': 'center', 'marginTop': 10, 'marginBottom': 5, 'fontSize': 15,
                               'fontWeight': 800, 'color': '#4B4D55'}),

                dcc.Graph(id='graph3', animate=True, config={'displayModeBar': False},
                          style={'marginLeft': 5, 'marginRight': 5, 'marginTop': 5}),

            ], className='nine columns',
                style={'background-color': 'white', 'height': '100%', 'border-width': 2, 'borderStyle': 'solid',
                       'border-radius': 10, 'borderColor': 'black'}),

        ], style={'height': '84%', 'position': 'relative', 'top': '1%'}),
    ], className='eight columns', style={'height': '100%'}),

    html.Div([
        html.Div([
            html.Div([
                html.H6('Roc-кривая',
                        style={'text-align': 'center', 'marginTop': 15, 'marginBottom': 5, 'fontSize': 15,
                               'fontWeight': 800}),
                dcc.Graph(id='roc-auc', config={'displayModeBar': False},
                          style={'marginLeft': 5, 'marginRight': 5}),
            ], style={'background-color': 'white', 'border-radius': 10, 'borderStyle': 'solid',
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
              [Input('dataset-choice', 'value'), Input('size-slider', 'value'), Input('noise-slider', 'value')])
def update_dataset(shape, size, noise):
    X, y = generate_data(size=size, shape=shape, noise=noise)
    X = json.dumps(X.tolist())
    y = json.dumps(y.tolist())
    return X, y


def generate_data(size, shape, noise):
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


@app.callback([Output('graph1', 'figure'),
               Output('graph2', 'figure'),
               Output('graph3', 'figure'), ],
              [Input('dataset-x', 'children'),
               Input('dataset-y', 'children'),
               Input('init', 'value'),
               Input('contamination', 'value')])
def create_plot(X, y, max_samples, contamination):
    X = np.array(json.loads(X))

    X = np.concatenate([X, np.random.RandomState(42).uniform(low=-6, high=6, size=(int(0.2 * 200), 2))], axis=0)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    is_forest = IsolationForest(max_samples=100, contamination=contamination, random_state=42)
    is_forest.fit(X)

    svm_one = svm.OneClassSVM(nu=contamination, kernel="rbf", gamma=0.1)
    svm_one.fit(X)

    copod = COPOD()
    copod.fit(X)
    y_pred = list()

    y_pred.append(is_forest.predict(X))
    y_pred.append(svm_one.predict(X))
    y_pred.append(copod.predict(X))

    Z_arr = list()
    Z = is_forest.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    Z_arr.append(Z)

    Z = svm_one.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    Z_arr.append(Z)

    Z = copod.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    Z_arr.append(Z)

    trace = viz_anomaly(X, y_pred[0], xx, yy, Z_arr[0])
    trace2 = viz_anomaly(X, y_pred[1], xx, yy, Z_arr[1])
    trace3 = viz_anomaly2(X, xx, yy, Z_arr[2])

    layout = go.Layout(
        hovermode='closest',
        legend=dict(x=0, y=-0.01, orientation="h"),
        margin=dict(l=0, r=0, t=0, b=0),
        height=250,
    )

    fig_if = go.Figure(data=trace, layout=layout)
    fig_svm = go.Figure(data=trace2, layout=layout)
    fig_copod = go.Figure(data=trace3, layout=layout)

    return fig_if, fig_svm, fig_copod


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

if __name__ == '__main__':
    app.run_server(debug=True)
