# pages/dataframes.py
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, callback
import pandas as pd
import plotly.express as px
import numpy as np


def create_layout(df):
    """Página DataFrames com foco visual"""

    return html.Div([
        html.H1(
            "📊 Análise de Dados",
            className="mb-3",
            style={"fontWeight": "600"}
        ),

        html.P(
            "Dashboard interativo com métricas, gráficos e perfil dos dados.",
            className="text-muted mb-4"
        ),

        # ===============================
        # CARDS PRINCIPAIS
        # ===============================
        dbc.Row([
            dbc.Col(_metric_card("Registros", f"{len(df):,}", "#0d6efd"), md=3),
            dbc.Col(_metric_card("Colunas", f"{len(df.columns)}", "#198754"), md=3),
            dbc.Col(_metric_card(
                "Valores Nulos",
                f"{df.isnull().sum().sum():,}",
                "#dc3545"
            ), md=3),
            dbc.Col(_metric_card(
                "Memória",
                f"{df.memory_usage(deep=True).sum()/1024**2:.1f} MB",
                "#fd7e14"
            ), md=3),
        ], className="mb-4"),

        # ===============================
        # TABS
        # ===============================
        dbc.Tabs([

            dbc.Tab(
                label="📌 Visão Geral",
                tab_id="tab-overview",
                children=[
                    html.Div(id="overview-content", className="mt-4")
                ]
            ),

            dbc.Tab(
                label="📋 Resumo Dataset",
                tab_id="tab-data",
                children=[
                    html.Div(id="data-content", className="mt-4")
                ]
            ),

            dbc.Tab(
                label="📊 Perfil Coluna",
                tab_id="tab-profile",
                children=[
                    html.Div(id="profile-content", className="mt-4")
                ]
            ),

            dbc.Tab(
                label="📈 Distribuições",
                tab_id="tab-dist",
                children=[
                    html.Div(id="dist-content", className="mt-4")
                ]
            ),

            dbc.Tab(
                label="🔗 Correlações",
                tab_id="tab-corr",
                children=[
                    html.Div(id="corr-content", className="mt-4")
                ]
            ),

        ], id="tabs", active_tab="tab-overview")

    ], className="p-4")


# =====================================
# CARD MÉTRICA
# =====================================

def _metric_card(title, value, color):
    return dbc.Card([
        dbc.CardBody([
            html.H6(title, className="text-muted"),
            html.H3(value, style={"color": color})
        ])
    ], className="shadow-sm border-0")


# =====================================
# CALLBACK PRINCIPAL
# =====================================

@callback(
    Output("overview-content", "children"),
    Output("data-content", "children"),
    Output("profile-content", "children"),
    Output("dist-content", "children"),
    Output("corr-content", "children"),
    Input("tabs", "active_tab")
)
def update_page(active_tab):
    from data_loader import data_manager
    df = data_manager.df

    # =================================
    # VISÃO GERAL
    # =================================
    nulls = df.isnull().sum().sort_values(ascending=False).head(10)

    fig_nulls = px.bar(
        x=nulls.index,
        y=nulls.values,
        title="Top 10 Colunas com Valores Nulos"
    )

    dtype_counts = df.dtypes.astype(str).value_counts()

    fig_types = px.pie(
        values=dtype_counts.values,
        names=dtype_counts.index,
        title="Tipos de Dados"
    )

    overview = dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_nulls), md=6),
        dbc.Col(dcc.Graph(figure=fig_types), md=6)
    ])

    # =================================
    # RESUMO DATASET
    # =================================
    data_tab = html.Div([

        html.H5("Resumo dos Dados", className="mb-4"),

        dbc.Row([
            dbc.Col(_metric_card("Linhas", f"{len(df):,}", "#0d6efd"), md=3),
            dbc.Col(_metric_card("Colunas", f"{len(df.columns)}", "#198754"), md=3),
            dbc.Col(_metric_card(
                "Nulos",
                f"{df.isnull().sum().sum():,}",
                "#dc3545"
            ), md=3),
            dbc.Col(_metric_card(
                "Duplicados",
                f"{df.duplicated().sum():,}",
                "#fd7e14"
            ), md=3),
        ], className="mb-4"),

        html.H6("Colunas disponíveis:"),

        html.Div([
            dbc.Badge(
                col,
                color="primary",
                className="me-2 mb-2 p-2"
            )
            for col in df.columns
        ])

    ])

    # =================================
    # PERFIL COLUNA
    # =================================
    profile = html.Div([
        dcc.Dropdown(
            id="column-selector",
            options=[
                {"label": c, "value": c}
                for c in df.columns
            ],
            value=df.columns[0]
        ),

        html.Div(id="column-profile", className="mt-4")
    ])

    # =================================
    # DISTRIBUIÇÕES
    # =================================
    numeric_cols = df.select_dtypes(include=np.number).columns

    dist_graphs = []

    for col in numeric_cols[:6]:
        fig = px.histogram(
            df,
            x=col,
            marginal="box",
            title=col
        )

        fig.update_layout(height=400)

        dist_graphs.append(
            dbc.Col(
                dcc.Graph(figure=fig),
                md=6
            )
        )

    dist_tab = dbc.Row(dist_graphs)

    # =================================
    # CORRELAÇÕES
    # =================================
    if len(numeric_cols) > 1:

        corr = df[numeric_cols].corr()

        fig_corr = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Mapa de Correlação"
        )

        fig_corr.update_layout(height=700)

        corr_tab = dcc.Graph(figure=fig_corr)

    else:
        corr_tab = dbc.Alert(
            "Sem colunas numéricas suficientes.",
            color="warning"
        )

    return overview, data_tab, profile, dist_tab, corr_tab


# =====================================
# PERFIL DE COLUNA
# =====================================

@callback(
    Output("column-profile", "children"),
    Input("column-selector", "value")
)
def update_profile(col):
    from data_loader import data_manager
    df = data_manager.df

    serie = df[col]

    # NUMÉRICA
    if pd.api.types.is_numeric_dtype(serie):

        fig = px.histogram(
            df,
            x=col,
            marginal="box",
            title=f"Distribuição de {col}"
        )

        cards = dbc.Row([
            dbc.Col(_metric_card("Média", f"{serie.mean():.2f}", "#0d6efd"), md=3),
            dbc.Col(_metric_card("Mediana", f"{serie.median():.2f}", "#198754"), md=3),
            dbc.Col(_metric_card("Mínimo", f"{serie.min():.2f}", "#dc3545"), md=3),
            dbc.Col(_metric_card("Máximo", f"{serie.max():.2f}", "#fd7e14"), md=3),
        ], className="mb-4")

        return html.Div([
            cards,
            dcc.Graph(figure=fig)
        ])

    # CATEGÓRICA
    else:

        top = serie.value_counts().head(10)

        fig = px.bar(
            x=top.index,
            y=top.values,
            title=f"Top Categorias - {col}"
        )

        return dcc.Graph(figure=fig)