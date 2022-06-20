import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import callbacks

from pages.main import kmeans
from pages.pca import pca
from pages.svm import svm
from pages.anomaly import anomaly

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.UNITED])
server = app.server

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@callback(Output('page-content', 'children'),
          [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/k-means':
        return kmeans()
    elif pathname == '/pca':
        return pca()
    elif pathname == '/svm':
        return svm()
    elif pathname == '/anomaly':
        return anomaly()
    else:
        return '404'


if __name__ == '__main__':
    app.run_server(debug=False)
