import json

import dash
import numpy as np
from dash import dcc, html
from dash.dependencies import Input, Output
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from svm_helper import generate_data, viz_svm, viz_roc_auc

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

                        ],
                        value='svm'
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
                                ],
                                value='moon'
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
                        ],  style={'marginLeft': 10, 'marginRight': 10}),

                    ]),

                    html.Div([
                        html.H6('Парамметры SVM',
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
                            html.Div('Параметр регуляризации(C) :', style={'marginLeft': 2}),
                            dcc.Slider(id='param_reg', min=0.1, max=1000, value=3,
                                       marks={0.1: '0.1', 1: '1', 10: '10', 100: '100', 1000: '1000'})
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
                html.H6('SVM график',
                        style={'text-align': 'center', 'marginTop': 30, 'marginBottom': 5, 'fontSize': 15,
                               'fontWeight': 800, 'color': '#4B4D55'}),

                dcc.Graph(id='svm-graph', animate=True, config={'displayModeBar': False},
                          style={'marginLeft': 5, 'marginRight': 15, 'marginTop': 30}),
              #  dcc.Interval(id='interval-component', interval=3600 * 1000, n_intervals=0)

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
              [Input('dataset-choice', 'value'), Input('size-slider', 'value'), Input('noise-slider', 'value')])
def update_dataset(shape, size, noise):
    X, y = generate_data(size=size, shape=shape, noise=noise)
    X = json.dumps(X.tolist())
    y = json.dumps(y.tolist())
    return X, y


@app.callback([Output('svm-graph', 'figure'), Output('roc-auc', 'figure')],
              [Input('dataset-x', 'children'),
               Input('dataset-y', 'children'),
               Input('init', 'value'),
               Input('param_reg', 'value'),
               Input('degree', 'value'),
               Input('gamma', 'value')])
def update_svm(X, y, kernel, C, degree, gamma):
    X = np.array(json.loads(X))
    y = np.array(json.loads(y))
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.3
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    clf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
    clf.fit(X_train, y_train)

    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    figure = viz_svm(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                     Z=Z, xx=xx, yy=yy, mesh_step=h)

    figure_roc = viz_roc_auc(clf, X_test, y_test)

    return figure, figure_roc


if __name__ == '__main__':
    app.run_server(debug=True)
