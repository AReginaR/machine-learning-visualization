import dash_bootstrap_components as dbc


def Navbar():
    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("K-means", href="/k-means")),
            dbc.NavItem(dbc.NavLink("PCA", href="/pca")),
            dbc.NavItem(dbc.NavLink("Anomaly", href="/anomaly")),
            dbc.NavItem(dbc.NavLink("SVM", href="/svm")),
        ],
    )
    return navbar
