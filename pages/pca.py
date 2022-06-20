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
                                id='dataset-choice2',
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
                        html.H6('Параметры PCA',
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
                        style={'text-align': 'center', 'marginTop': 5, 'marginBottom': 5, 'fontSize': 15,
                               'fontWeight': 800}),
                dcc.Graph(id='pca-graph', animate=True, config={'displayModeBar': False},
                          style={'marginLeft': 5, 'marginRight': 5, 'marginTop': 5}),
                html.H6('Результат алгоритма',
                        style={'text-align': 'center', 'marginTop': 5, 'marginBottom': 5, 'fontSize': 15,
                               'fontWeight': 800}),
                dcc.Graph(id='pca-graph-res', animate=True, config={'displayModeBar': False},
                          style={'marginLeft': 5, 'marginRight': 5, 'marginTop': 5}),

            ], className='nine columns',
                style={'background-color': 'white', 'height': '100%', 'border-width': 2, 'borderStyle': 'solid',
                       'border-radius': 10, 'borderColor': 'black'}),

        ], style={'height': '80%', 'position': 'relative', 'top': '1%'}),
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

        ], style={'height': '80%', 'position': 'relative', 'top': '1%'}),
    ], className='four columns', style={'height': '100%', 'marginLeft': '1.5%', 'width': '32.5%', 'top': '1%'}),

    html.Div([
        html.Div(id='dataset-x', style={'display': 'none'}),
        html.Div(id='dataset-y', style={'display': 'none'}),
        html.Div(id='svm_counter', style={'display': 'none'}),
        html.Div(id='output-data-upload'),
    ], className='twelve columns', style={'marginTop': '1%'})

], style={'height': 820})


def pca():
    layout = html.Div([
        nav,
        body
    ])
    return layout


if __name__ == '__main__':
    app.run_server(debug=True)
