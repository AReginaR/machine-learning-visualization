import base64
import io
import json

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import dash_table, callback
from dash import html
from dash.dependencies import Input, Output
from dash.dependencies import State
from pyod.models.copod import COPOD
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from helpers.anomaly_helper import viz_anomaly, viz_anomaly2, generate_data_anomaly
from helpers.helper import generate_data, generate_centroids, expectation, maximization, viz, k_means_score
from helpers.pca_helper import generate_data_pca, get_viz_data__, get_viz_res, get_viz_data, viz_bar, cov_mat
from helpers.svm_helper import generate_data_svm, viz_svm, viz_roc_auc

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


@callback(Output('dataset', 'children'),
          [Input('dataset-choice', 'value'), Input('size', 'value'), Input('cluster-count', 'value'), Input('output-data-upload', 'children')])
def create_data(sample_shape, sample_size, n_clusters, up):
    data = generate_data(shape=sample_shape, size=sample_size, n_clusters=n_clusters)
    data = json.dumps(data.tolist())
    return data


@callback(Output('centroids-value', 'children'),
          [Input('init-method', 'value'), Input('centroid-count', 'value'), Input('dataset', 'children')])
def create_centroids(init_method, k, dataset):
    data = np.array(json.loads(dataset))
    centroids = generate_centroids(data, k=k, init_method=init_method)
    centroids = json.dumps(centroids.tolist())
    return centroids


@callback(Output('kmeans-counter', 'children'),
          [Input('dataset', 'children'), Input('centroids-value', 'children'),
           Input('max_iter', 'value'), Input('output-data-upload', 'children')])
def create_k_means(dataset, kmeans_centroids, max_iter, up):
    global count_k_means, global_frames, global_silhouette, global_sse, silhouette_traces

    count_k_means = count_k_means + 1
    if up:
        data = pd.read_csv('data.csv', index_col='id')
        data = np.array(data)
        print(data)
    else:
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


@callback(Output('kmeans-graph', 'figure'),
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


@callback(Output('iter-text', 'children'), [Input('kmeans-graph', 'figure')])
def update_iter_text(kmeans_fig):
    text = global_frames[global_current_step]['layout']['title']
    return text


@callback(Output('interval-component', 'interval'),
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


@callback([Output('elbow-method', 'figure'), Output('text-param-elbow', 'children')],
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
        'Предпочтительное количество кластеров по методу локтя: {}'.format(3))


@callback([Output('silhouette-method', 'figure'), Output('text-param-silh', 'children')],
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


@callback([Output('dataset-x', 'children'), Output('dataset-y', 'children')],
          [Input('dataset-choice2', 'value'), Input('size-slider', 'value')])
def update_dataset_pca(shape, size):
    X, y = generate_data_pca(size=size, shape=shape)
    X = json.dumps(X.tolist())
    y = json.dumps(y.tolist())
    return X, y


covmat = list()
color = {0: 'deepskyblue', 1: 'lightcoral', 2: 'limegreen', 3: 'gold', 4: 'plum', 5: 'lightpink',
         6: 'mediumpurple', 7: 'chocolate', 8: 'orange'}


@callback([Output('pca-graph', 'figure'),
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

        layout4 = go.Layout(
            hovermode='closest',
            legend=dict(x=0, y=-0.01, orientation="h"),
            margin=dict(l=0, r=0, t=0, b=0),
            height=300,
        )

        fig = get_viz_res(X_new, y, loadings, text)

        return go.Figure(data=trace, layout=layout4), fig, fig_var, fig_cov

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


@callback([Output('dataset-x-svm', 'children'), Output('dataset-y-svm', 'children')],
          [Input('dataset-choice3', 'value'), Input('size-slider3', 'value'), Input('noise-slider3', 'value')])
def update_dataset_svm(shape, size, noise):
    X, y = generate_data_svm(size=size, shape=shape, noise=noise)
    X = json.dumps(X.tolist())
    y = json.dumps(y.tolist())
    return X, y


@callback([Output('svm-graph', 'figure'), Output('roc-auc', 'figure')],
          [Input('dataset-x-svm', 'children'),
           Input('dataset-y-svm', 'children'),
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


@callback([Output('dataset-x-a', 'children'), Output('dataset-y-a', 'children')],
          [Input('dataset-choice4', 'value'), Input('size-slider4', 'value'), Input('noise-slider4', 'value')])
def update_dataset(shape, size, noise):
    X, y = generate_data_anomaly(size=size, shape=shape, noise=noise)
    X = json.dumps(X.tolist())
    y = json.dumps(y.tolist())
    return X, y


@callback([Output('graph1', 'figure'),
           Output('graph2', 'figure'),
           Output('graph3', 'figure'), ],
          [Input('dataset-x-a', 'children'),
           Input('dataset-y-a', 'children'),
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
        data = df.to_json(orient='values')
        #data = json.loads
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return data


@callback(Output('output-data-upload', 'children'),
          Input('upload-data', 'contents'),
          State('upload-data', 'filename'),
          State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        content_type, content_string = list_of_contents[0].split(',')

        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in list_of_names[0]:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')))
            elif 'xls' in list_of_names[0]:
                # Assume that the user uploaded an excel file
                df = pd.read_excel(io.BytesIO(decoded))
            data = df.to_json(orient='values')
            # data = json.loads
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])
      #  print(data)
        children = json.dumps(data)
        return children


@callback([Output('cluster-count', 'disabled'),
          Output('dataset-choice', 'disabled'),
          Output('size', 'disabled')],
          [Input('dataset-choice', 'value'),
           Input('output-data-upload', 'children')])
def disable_component(shape, up):
    if up:
        return True, True, True
    if shape in ['moon', 'circle', 'noStructure']:
        return True, False, False
    return False, False, False
