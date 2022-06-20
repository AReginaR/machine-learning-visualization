import dash
from dash import dcc, html

from navbar import Navbar

app = dash.Dash(__name__)

nav = Navbar()

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
                                id='dataset-choice4',
                                options=[
                                    {'label': 'Линейно', 'value': 'linear'},
                                    {'label': 'Концентрические круги', 'value': 'circle'},
                                    {'label': 'Луны', 'value': 'moon'},
                                    {'label': 'Blobs', 'value': 'blobs'},
                                ],
                                value='circle'
                            )
                        ], style={'marginLeft': 10, 'marginRight': 10, 'marginBottom': 5}),

                        html.Div([
                            html.Div('Размер :', style={'marginLeft': 2}),
                            dcc.Slider(id='size-slider4', min=50, max=500, step=50, value=200,
                                       marks={50: '50', 500: '500'})
                        ], style={'marginLeft': 10, 'marginRight': 10, 'marginBottom': 5}),

                        html.Div([
                            html.Div('Шум :', style={'marginLeft': 2}),
                            dcc.Slider(id='noise-slider4', min=0.0, max=1.0, step=0.1,
                                       marks={i / 10: str(i / 10) for i in range(0, 11, 2)},
                                       value=.1)
                        ], style={'marginLeft': 10, 'marginRight': 10}),

                    ]),

                    html.Div([
                        html.H6('Параметры ',
                                style={'text-align': 'center', 'marginBottom': 5, 'fontSize': 15, 'fontWeight': 800}),

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
                        ], style={'marginLeft': 10, 'marginRight': 10, 'marginBottom': 5}),

                        html.Div([
                            html.Div('процент аномалий :', style={'marginLeft': 2}),
                            dcc.Slider(id='contamination', min=0.1, max=1, value=0.2, step=0.1,
                                       marks={0.2: '0.2', 0.4: '0.4', 0.6: '0.6'})
                        ], style={'marginLeft': 10, 'marginRight': 10, 'marginBottom': 5}),

                        html.H6('Iforest',
                                style={'text-align': 'center', 'marginBottom': 10, 'fontSize': 15, 'fontWeight': 800}),

                        html.Div([
                            html.Div('max samples :', style={'marginLeft': 2}),
                            dcc.Slider(id='max_samples', min=10, max=100, step=10, value=30)
                        ], style={'marginLeft': 10, 'marginRight': 10, 'marginBottom': 5}),

                        html.Div([
                            html.Div('n_estimators  :', style={'marginLeft': 2}),
                            dcc.Slider(id='n_estimators ', min=100, max=800, value=100,step=50)
                        ], style={'marginLeft': 10, 'marginRight': 10, 'marginBottom': 5}),

                        html.H6('One-Class SVM',
                                style={'text-align': 'center', 'marginBottom': 5, 'fontSize': 15, 'fontWeight': 800}),

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
                        ], style={'marginLeft': 10, 'marginRight': 10, 'marginBottom': 5}),

                        html.Div([
                            html.Div('gamma :', style={'marginLeft': 2}),
                            dcc.Slider(id='gamma', min=0.1, max=100, value=1,
                                       marks={0.1: '0.1', 1: '1', 10: '10', 100: '100'})
                        ], style={'marginLeft': 10, 'marginRight': 10, 'marginBottom': 5}),

                    ]),
                ]),

            ], className='three columns',
                style={'background-color': 'white', 'borderStyle': 'solid', 'border-radius': 10,
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

        ], style={'position': 'relative', 'top': '1%'}),
    ], className='eight columns', style={'height': '150%'}),

    html.Div([
        html.Div(id='dataset-x-a', style={'display': 'none'}),
        html.Div(id='dataset-y-a', style={'display': 'none'}),
        html.Div(id='svm_counter', style={'display': 'none'}),
        html.Div(id='output-data-upload'),
    ], className='twelve columns', style={'marginTop': '1%'})

], style={'height': 820})


def anomaly():
    layout = html.Div([
        nav,
        body
    ])
    return layout


if __name__ == '__main__':
    app.run_server(debug=True)
