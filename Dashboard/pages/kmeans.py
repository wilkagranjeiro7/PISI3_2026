# ==============================================
# 📁 pages/kmeans.py (VERSÃO ESTÁVEL - SEM LOOP)
# ==============================================

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import numpy as np


# ==============================================
# 🎯 LAYOUT
# ==============================================
def create_layout(df=None):

    default_features = ['recovery_score', 'day_strain']

    if df is not None and not df.empty:
        available_features = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col not in ['id', 'day']
        ][:5]
    else:
        available_features = default_features

    return html.Div([
        html.H1("🎯 K-Means Dashboard", className="mb-4"),

        html.Div(id='kmeans-alert'),

        dbc.Row([
            # CONTROLES
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("⚙️ Configurações"),
                    dbc.CardBody([

                        html.Label("K"),
                        dcc.Slider(
                            id='kmeans-k',
                            min=2, max=8, step=1,
                            value=3,
                            marks={i: str(i) for i in range(2, 9)}
                        ),

                        html.Hr(),

                        html.Label("Features"),
                        dcc.Dropdown(
                            id='kmeans-features',
                            multi=True,
                            value=default_features,
                            options=[
                                {'label': f.replace('_', ' ').title(), 'value': f}
                                for f in available_features
                            ]
                        ),

                        html.Hr(),

                        html.Label("X"),
                        dcc.Dropdown(id='kmeans-x'),

                        html.Br(),

                        html.Label("Y"),
                        dcc.Dropdown(id='kmeans-y'),

                        html.Hr(),

                        dbc.Button(
                            "Executar",
                            id="kmeans-run-btn",
                            color="primary",
                            className="w-100"
                        )
                    ])
                ])
            ], width=3),

            # RESULTADO
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Resultados"),
                    dbc.CardBody([
                        html.Div(id='kmeans-stats'),
                        dcc.Graph(id='kmeans-graph'),
                        dcc.Graph(id='elbow-graph')
                    ])
                ])
            ], width=9)
        ])
    ])


# ==============================================
# 🔁 CALLBACKS
# ==============================================
def register_kmeans_callbacks(app, df_provider):

    # ===============================
    # DROPDOWNS (SEM LOOP)
    # ===============================
    @app.callback(
        Output('kmeans-x', 'options'),
        Output('kmeans-x', 'value'),
        Output('kmeans-y', 'options'),
        Output('kmeans-y', 'value'),
        Input('kmeans-features', 'value')
    )
    def update_axes(features):

        if not features:
            raise PreventUpdate

        options = [
            {'label': f.replace('_', ' ').title(), 'value': f}
            for f in features
        ]

        x = features[0]
        y = features[1] if len(features) > 1 else features[0]

        return options, x, options, y


    # ===============================
    # KMEANS PRINCIPAL
    # ===============================
    @app.callback(
        Output('kmeans-stats', 'children'),
        Output('kmeans-graph', 'figure'),
        Output('elbow-graph', 'figure'),
        Output('kmeans-alert', 'children'),
        Input('kmeans-run-btn', 'n_clicks'),
        State('kmeans-k', 'value'),
        State('kmeans-features', 'value'),
        prevent_initial_call=True
    )
    def run_kmeans(n_clicks, k, features):

        df = df_provider() if callable(df_provider) else df_provider

        if df is None or df.empty:
            raise PreventUpdate

        if not features or len(features) < 2:
            fig = go.Figure()
            fig.update_layout(title="Selecione pelo menos 2 features")

            return (
                html.Div("Selecione features"),
                fig,
                fig,
                dbc.Alert("Selecione pelo menos 2 features", color="warning")
            )

        try:
            # ===============================
            # PREPARAÇÃO
            # ===============================
            X = df[features].copy()
            X = X.fillna(X.median(numeric_only=True))

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # ===============================
            # KMEANS
            # ===============================
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)

            silhouette = (
                silhouette_score(X_scaled, clusters)
                if len(np.unique(clusters)) > 1 else 0
            )

            # ===============================
            # PCA
            # ===============================
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            centers = pca.transform(kmeans.cluster_centers_)

            # ===============================
            # ELBOW
            # ===============================
            k_range = range(2, min(9, len(df)))
            wcss = []

            for i in k_range:
                km = KMeans(n_clusters=i, random_state=42, n_init=10)
                km.fit(X_scaled)
                wcss.append(km.inertia_)

            # ===============================
            # FIGURAS
            # ===============================
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=clusters,
                    colorscale='Viridis',
                    showscale=True
                ),
                text=[f"Cluster {c}" for c in clusters],
                hovertemplate="Cluster: %{text}<extra></extra>"
            ))

            fig.add_trace(go.Scatter(
                x=centers[:, 0],
                y=centers[:, 1],
                mode='markers+text',
                marker=dict(size=14, color='red', symbol='x'),
                text=[f"C{i}" for i in range(k)]
            ))

            fig.update_layout(
                title="Clusters (PCA)",
                template="plotly_white",
                height=500
            )

            # ELBOW
            fig_elbow = go.Figure()

            fig_elbow.add_trace(go.Scatter(
                x=list(k_range),
                y=wcss,
                mode='lines+markers'
            ))

            fig_elbow.update_layout(
                title="Elbow Method",
                template="plotly_white"
            )

            # ===============================
            # STATS
            # ===============================
            stats = html.Div([
                html.H5(f"K = {k}"),
                html.P(f"Silhouette: {silhouette:.3f}")
            ])

            return (
                stats,
                fig,
                fig_elbow,
                None
            )

        except Exception as e:
            fig = go.Figure()
            fig.update_layout(title=str(e))

            return (
                html.Div(str(e)),
                fig,
                fig,
                dbc.Alert(str(e), color="danger")
            )