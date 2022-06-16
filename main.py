import base64
import datetime
import io
import json

import dash
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from sklearn.preprocessing import StandardScaler

from helper import generate_data, generate_centroids, expectation, maximization, viz, k_means_score

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
                            {'label': 'PCA', 'value': 'pca'},
                            {'label': 'Gradient', 'value': 'grad'},
                        ],
                        value='k-means'
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
                ], id='kmeans_form', style={'overflow': 'auto'}),

            ], className='three columns',
                style={'background-color': 'white', 'height': '100%', 'borderStyle': 'solid', 'border-radius': 10,
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
                       'border-radius': 10, 'borderColor': 'black'}),

        ], style={'height': '84%', 'position': 'relative', 'top': '1%'}),
    ], className='eight columns', style={'height': '100%'}),

    html.Div([
        html.Div([
            html.Div([
                html.H6('Метод локтя',
                        style={'text-align': 'center', 'marginTop': 15, 'marginBottom': 5, 'fontSize': 15,
                               'fontWeight': 800}),
                dcc.Graph(id='elbow-method', config={'displayModeBar': False},
                          style={'marginLeft': 5, 'marginRight': 5}),
            ], style={'background-color': 'white', 'height': '49%', 'border-radius': 10, 'borderStyle': 'solid',
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
        html.Div(id='output-data-upload'),
    ], className='twelve columns', style={'marginTop': '1%'})
], style={'height': 820})

count_k_means = -1
global_frames = list()
global_sse = list()
global_silhouette = list()
global_current_step = 0
global_prev_clicks = 0
global_next_clicks = 0
global_restart_clicks = 0
global_num_intervals = 0
global_frames_counter = 0
global_play_clicks = 0
global_pause_clicks = 0
silhouette_traces = list()


@app.callback(Output('dataset', 'children'),
              [Input('dataset-choice', 'value'), Input('size', 'value'), Input('cluster-count', 'value')])
def create_data(sample_shape, sample_size, n_clusters):
    data = generate_data(shape=sample_shape, size=sample_size, n_clusters=n_clusters)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data = json.dumps(data.tolist())
    return data


@app.callback(Output('centroids-value', 'children'),
              [Input('init-method', 'value'), Input('centroid-count', 'value'), Input('dataset', 'children')])
def create_centroids(init_method, k, dataset):
    data = np.array(json.loads(dataset))
    centroids = generate_centroids(data, k=k, init_method=init_method)
    centroids = json.dumps(centroids.tolist())
    return centroids


@app.callback(Output('kmeans-counter', 'children'),
              [Input('dataset', 'children'), Input('centroids-value', 'children'),
               Input('max_iter', 'value')])
def create_k_means(dataset, kmeans_centroids, max_iter):
    global count_k_means, global_frames, global_silhouette, global_sse, silhouette_traces

    count_k_means = count_k_means + 1

    data = np.array(json.loads(dataset))
    centroids = np.array(json.loads(kmeans_centroids))
    labels = [-1] * data.shape[0]

    kmeans_frames = [viz(data, labels, centroids)]
    for i in range(max_iter):
        labels = expectation(data, centroids)
        kmeans_frames.append(viz(data, labels, centroids))

        centroids = maximization(data, centroids, labels)
        kmeans_frames.append(viz(data, labels, centroids))

    global_sse, global_silhouette, silhouette_traces = k_means_score(data)

    global_frames = [{'data': kmeans_frames[0], 'layout': {**layout, 'title': 'Инициализация...'}}]
    global_frames = global_frames + [
        {'data': d, 'layout': {**layout, 'title': 'Шаг {} : {}'.format(idx // 2 + 1, 'Ожидание')}} if idx % 2 == 0
        else {'data': d, 'layout': {**layout, 'title': 'Шаг {} : {}'.format(idx // 2 + 1, 'Максимизация')}}
        for idx, d in enumerate(kmeans_frames[1:])]

    return count_k_means


layout = dict(
    xaxis=dict(zeroline=False, showgrid=False, showline=True, showticklabels=False, ),
    yaxis=dict(zeroline=False, showgrid=False, showline=True, showticklabels=False, ),
    hovermode='closest',
    margin={'t': 10, 'b': 10, 'l': 10},
    legend=dict(x=0, y=-0.01, orientation="h"),
)


@app.callback(Output('kmeans-graph', 'figure'),
              [
                  Input('nextStep-button', 'n_clicks'), Input('prevStep-button', 'n_clicks'),
                  Input('restart-button', 'n_clicks_timestamp'),
                  Input('kmeans-counter', 'children'), Input('interval-component', 'n_intervals')
              ])
def update_graph(next_step_n_clicks, prev_step_n_clicks, restart_n_clicks, frames_counter, n_intervals):
    global global_prev_clicks, global_next_clicks, global_current_step, global_restart_clicks, global_num_intervals, \
        global_frames_counter

    if prev_step_n_clicks is None:
        prev_step_n_clicks = 0
    if next_step_n_clicks is None:
        next_step_n_clicks = 0
    if restart_n_clicks is None:
        restart_n_clicks = 0
    if n_intervals is None:
        n_intervals = 0

    if (global_restart_clicks != restart_n_clicks) or (global_frames_counter != frames_counter):
        global_restart_clicks = restart_n_clicks
        global_frames_counter = frames_counter
        global_current_step = 0
        d = global_frames[global_current_step]['data']
        fig = go.Figure(data=d, layout=layout)
        return fig

    elif (global_next_clicks != next_step_n_clicks) or (global_num_intervals != n_intervals):
        global_next_clicks = next_step_n_clicks
        global_num_intervals = n_intervals
        global_current_step = min(global_current_step + 1, len(global_frames) - 1)
        d = global_frames[global_current_step]['data']
        fig = go.Figure(data=d, layout=layout)
        return fig

    elif global_prev_clicks != prev_step_n_clicks:
        global_prev_clicks = prev_step_n_clicks
        global_current_step = max(global_current_step - 1, 0)
        d = global_frames[global_current_step]['data']
        fig = go.Figure(data=d, layout=layout)
        return fig

    d = global_frames[global_current_step]['data']
    fig = go.Figure(data=d, layout=layout)
    return fig


@app.callback(Output('iter-text', 'children'), [Input('kmeans-graph', 'figure')])
def update_iter_text(kmeans_fig):
    text = global_frames[global_current_step]['layout']['title']
    return text


@app.callback(Output('cluster-count', 'disabled'), [Input('dataset-choice', 'value')])
def disable_component(shape):
    if shape in ['moon', 'circle', 'noStructure']:
        return True
    return False


@app.callback(Output('interval-component', 'interval'),
              [Input('play-button', 'n_clicks'), Input('pause-button', 'n_clicks')])
def play_kmeans(play_clicks, pause_clicks):
    global global_play_clicks, global_pause_clicks
    if play_clicks is None:
        play_clicks = 0
    if pause_clicks is None:
        pause_clicks = 0
    if global_play_clicks != play_clicks:
        global_play_clicks = play_clicks
        return 1000
    return 3600 * 1000


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])


@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


@app.callback([Output('elbow-method', 'figure'), Output('text-param-elbow', 'children')],
              [Input('kmeans-counter', 'children')])
def update_elbow_graph(frames_counter):
    data = go.Scatter(
        x=[i for i in range(2, 10)],
        y=global_sse,
        mode='markers+lines',
        marker=dict(color='white', size=10, line=dict(width=2, color='#D69D9D')),
        line=dict(color='#D69D9D'),
    )

    param_elbow_method = go.Layout(
        xaxis=dict(title='Кластеры', zeroline=False, showgrid=False, showline=True, mirror='ticks'),
        yaxis=dict(title='Оценка', zeroline=False, showgrid=True, showline=True, mirror='ticks', gridcolor="silver", ),
        height=250,
        margin={'t': 10, 'l': 60, 'r': 10, 'b': 40},
        showlegend=False,
    )

    return go.Figure(data=data, layout=param_elbow_method), html.P(
        'Предпочтительное количество кластеров по методу локтя: {}'.format(0))


@app.callback([Output('silhouette-method', 'figure'), Output('text-param-silh', 'children')],
              [Input('kmeans-counter', 'children'), Input('sil-option', 'value')])
def update_sil_graph(kmeans_counter, counter):
    max_index = np.argmax(global_silhouette)
    if int(counter) == 1:

        data = go.Scatter(
            y=global_silhouette,
            x=[i for i in range(2, 10)],
            mode='markers+lines',
            marker=dict(color='white', size=10, line=dict(width=2, color='#D69D9D')),
            line=dict(color='#D69D9D'),
        )

        param_silhoette_method = go.Layout(
            xaxis=dict(title='Кластеры', zeroline=False, showgrid=False, showline=True, mirror='ticks'),
            yaxis=dict(title='Коэффициэнт', zeroline=False, showgrid=True, showline=True, mirror='ticks',
                       gridcolor="silver", ),
            height=240,
            margin={'t': 10, 'l': 60, 'r': 10, 'b': 40},
            showlegend=False,
        )

        return go.Figure(data=data, layout=param_silhoette_method), html.P(
            'Предпочтительное количество кластеров по методу силуэта: {}'.format(max_index + 2))

    else:
        n = int(counter) - 2

        layout = dict(
            xaxis=dict(title='Коэффицент силуэта', zeroline=False, showgrid=True, showline=True, linewidth=2,
                       mirror='ticks', gridcolor="silver"),
            yaxis=dict(title='Кластер', showticklabels=False, zeroline=False, showgrid=True, showline=True,
                       linewidth=2, mirror='ticks'),
            height=245,
            margin={'t': 10, 'l': 60, 'r': 10, 'b': 40},
            showlegend=False,
        )

        return go.Figure(data=silhouette_traces[n], layout=layout), html.P(
            'Предпочтительное количество кластеров по методу силуэта: {}'.format(max_index + 2))


if __name__ == '__main__':
    app.run_server(debug=True)
