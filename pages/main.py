import dash
from dash import dcc, html

from navbar import Navbar

nav = Navbar()

app = dash.Dash(__name__)

body = html.Div(children=[
    html.Div([
        html.Div([
            html.Div([

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
                                    {'label': 'Гауссово распределение', 'value': 'gmm'},
                                    {'label': 'Концентрические круги', 'value': 'circle'},
                                    {'label': 'Луны', 'value': 'moon'},
                                    {'label': 'Анизотропно распределенный', 'value': 'anisotropic'},
                                    {'label': 'Без структуры', 'value': 'noStructure'},
                                ],
                                value='gmm')
                        ], style={'marginLeft': 10, 'marginRight': 10, 'marginBottom': 10}),

                        html.Div([
                            html.Div('Размер :', style={'marginLeft': 2}),
                            dcc.Slider(id='size', min=50, max=500, step=50, value=200,
                                       marks={50: '50', 500: '500'})
                        ], style={'marginLeft': 10, 'marginRight': 10, 'marginBottom': 10}),

                        html.Div([
                            html.Div('Количество кластеров :', style={'marginLeft': 2}),
                            dcc.Slider(id='cluster-count', min=2, max=9,
                                       marks={i: '{}'.format(i) for i in range(2, 10)},
                                       value=3)
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
                        ]),

                    ]),

                    html.Hr(style={'marginTop': 10, 'marginBottom': 10}),

                    html.Div([
                        html.H6('Иниациализация k-means',
                                style={'text-align': 'center', 'marginBottom': 10, 'fontSize': 15, 'fontWeight': 800}),

                        html.Div([
                            html.Div('Метод инициализации :', style={'marginLeft': 2}),
                            dcc.Dropdown(
                                id='init-method',
                                options=[
                                    {'label': 'Рандом', 'value': 'random'},
                                    {'label': 'K-means++', 'value': 'kmeans++'}
                                ],
                                value='random'
                            )
                        ], style={'marginLeft': 10, 'marginRight': 10, 'marginBottom': 10}),

                        html.Div([
                            html.Div('Количество кластеров :', style={'marginLeft': 2}),
                            dcc.Slider(id='centroid-count',
                                       min=2,
                                       max=9,
                                       marks={i: '{}'.format(i) for i in range(2, 10)},
                                       value=3)
                        ], style={'marginLeft': 10, 'marginRight': 10, 'marginBottom': 10}),

                        html.Div([
                            html.Div('Максимум итераций :', style={'marginLeft': 2}),
                            dcc.Slider(id='max_iter', min=5, max=20, step=1, value=10, marks={5: '5', 20: '20'})
                        ], style={'marginLeft': 10, 'marginRight': 10, 'marginBottom': 10}),

                    ]),
                ], id='kmeans_form'),

            ], className='three columns',
                style={'background-color': 'white', 'height': '100%', 'borderStyle': 'solid', 'border-radius': 5,
                       'borderWidth': 2, 'borderColor': 'black'}),

            html.Div([
                html.H4('график K-means', id='nameGraph',
                        style={'text-align': 'center', 'marginTop': 30, 'marginBottom': 5, 'fontSize': 15,
                               'fontWeight': 800}),

                html.Button('Пуск', id='play-button',
                            style={'marginLeft': 45, 'marginTop': 30, 'marginBottom': 10, 'marginRight': 5,
                                   'width': '8%', 'font-size': 10, 'text-align': 'center', 'padding': 0}),
                html.Button('Пауза', id='pause-button',
                            style={'marginRight': 5, 'width': '8%', 'font-size': 10, 'text-align': 'center',
                                   'padding': 0}),
                html.Button('<<', id='prevStep-button',
                            style={'marginRight': 5, 'width': '5%', 'font-size': 10, 'text-align': 'center',
                                   'padding': 0}),
                html.Button('>>', id='nextStep-button',
                            style={'marginRight': 5, 'width': '5%', 'font-size': 10, 'text-align': 'center',
                                   'padding': 0}),
                html.Button('Заново', id='restart-button',
                            style={'width': '10%', 'font-size': 10, 'text-align': 'center', 'padding': 0}),

                html.Button(id='iter-text', disabled=True,
                            style={'marginLeft': '24%', 'width': '20%', 'background-color': '#8fcfc8',
                                   'pointer-events': 'none', 'color': 'white', 'font-size': 10, 'text-align': 'center',
                                   'padding': 0}),

                dcc.Graph(id='kmeans-graph', animate=True, config={'displayModeBar': False}),

                dcc.Interval(id='interval-component', interval=3600 * 1000, n_intervals=0),

                html.Div(id='text-param-elbow', style={'marginLeft': 40}),

                html.Div(id='text-param-silh', style={'marginLeft': 40})

            ], className='nine columns',
                style={'background-color': 'white', 'height': '100%', 'border-width': 2, 'borderStyle': 'solid',
                       'border-radius': 5, 'borderColor': 'black'}),

        ], style={'height': '80%', 'position': 'relative', 'top': '1%'}),
    ], className='eight columns', style={'height': '100%'}),

    html.Div([
        html.Div([
            html.Div([
                html.H6('Метод локтя',
                        style={'text-align': 'center', 'marginTop': 15, 'marginBottom': 5, 'fontSize': 15,
                               'fontWeight': 800}),
                dcc.Graph(id='elbow-method', config={'displayModeBar': False},
                          style={'marginLeft': 5, 'marginRight': 5}),
            ], style={'background-color': 'white', 'height': '45%', 'border-radius': 10, 'borderStyle': 'solid',
                      'borderWidth': 2, 'borderColor': 'black'}),

            html.Div([
                html.H6('Силуэт',
                        style={'text-align': 'center', 'marginTop': 15, 'marginBottom': 5, 'fontSize': 15,
                               'fontWeight': 800, 'color': '#4B4D55'}),
                dcc.Dropdown(
                    id='sil-option',
                    options=[
                        {'label': 'Общий', 'value': '1'},
                        {'label': 'для 2 кластеров', 'value': '2'},
                        {'label': 'для 3 кластеров', 'value': '3'},
                        {'label': 'для 4 кластеров', 'value': '4'},
                        {'label': 'для 5 кластеров', 'value': '5'},
                        {'label': 'для 6 кластеров', 'value': '6'},
                        {'label': 'для 7 кластеров', 'value': '7'},
                        {'label': 'для 8 кластеров', 'value': '8'},
                        {'label': 'для 9 кластеров', 'value': '9'},
                    ],
                    value='1',
                    style={'marginLeft': 10, 'marginRight': 20, 'marginBottom': 10},
                ),
                dcc.Graph(id='silhouette-method', config={'displayModeBar': False},
                          style={'marginLeft': 5, 'marginRight': 5})
            ], style={'background-color': 'white', 'height': '49%', 'position': 'relative', 'top': '1.6%',
                      'border-radius': 10, 'borderStyle': 'solid', 'borderWidth': 2, 'borderColor': 'black'}),
        ], style={'height': '84%', 'position': 'relative', 'top': '1%'}),
    ], className='four columns', style={'height': '100%', 'marginLeft': '1.5%', 'width': '32.5%', 'top': '1%'}),

    html.Div([
        html.Div(id='dataset', style={'display': 'none'}),
        html.Div(id='centroids-value', style={'display': 'none'}),
        html.Div(id='kmeans-counter', style={'display': 'none'}),
        html.Div(id='output-data-upload', style={'display': 'none'}),
    ], className='twelve columns', style={'marginTop': '1%'})
], style={'height': 820})


def kmeans():
    layout = html.Div([
        nav,
        body
    ])
    return layout


if __name__ == '__main__':
    app.run_server(debug=True)
