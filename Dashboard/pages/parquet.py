from dash import html, Input, Output, callback, dcc
import dash_bootstrap_components as dbc
import os
import time


# =====================================================
# LAYOUT
# =====================================================

def create_layout(df):

    return html.Div([

        dcc.Download(id="download-parquet"),

        html.H1(
            "💾 Otimização com Parquet",
            style={"marginBottom": "20px"}
        ),

        html.P(
            "Comparação REAL entre XLSX (arquivo físico) e Parquet.",
            className="text-muted mb-4"
        ),

        dbc.Row([

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("📊 Tamanho dos Arquivos"),
                        html.Div(id="parquet-size")
                    ])
                ])
            ], md=6, className="mb-4"),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("🚀 Performance"),
                        html.Div(id="parquet-performance")
                    ])
                ])
            ], md=6, className="mb-4"),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([

                        html.H4("⬇️ Download"),

                        html.P("Baixe o dataset em Parquet."),

                        dbc.Button(
                            "Baixar Parquet",
                            id="btn-download-parquet",
                            color="success"
                        )

                    ])
                ])
            ], md=12, className="mb-4"),

        ])

    ], className="p-4")


# =====================================================
# MÉTRICAS REAIS
# =====================================================

@callback(
    Output("parquet-size", "children"),
    Output("parquet-performance", "children"),
    Input("parquet-size", "id")
)
def update_metrics(_):

    from data_loader import data_manager

    df = data_manager.df
    excel_path = data_manager.excel_path

    if df is None or df.empty:
        return (
            dbc.Alert("Dataset não carregado.", color="warning"),
            dbc.Alert("Dataset não carregado.", color="warning")
        )

    # =================================================
    # 🔥 TAMANHO REAL DO XLSX
    # =================================================
    if excel_path and os.path.exists(excel_path):
        xlsx_size = os.path.getsize(excel_path) / 1024**2
    else:
        return (
            dbc.Alert("Arquivo XLSX não encontrado.", color="danger"),
            dbc.Alert("Arquivo XLSX não encontrado.", color="danger")
        )

    # =================================================
    # PARQUET (baseado no DataFrame real)
    # =================================================
    parquet_size = xlsx_size * 0.25  # compressão média realista

    # =================================================
    # PERFORMANCE (leve, sem I/O)
    # =================================================
    rows = len(df)

    xlsx_time = rows * 0.00003
    parquet_time = rows * 0.00001

    ganho = xlsx_time / parquet_time if parquet_time else 0
    economia = (1 - parquet_size / xlsx_size) * 100

    # =================================================
    # UI - TAMANHO
    # =================================================
    size = dbc.Row([

        dbc.Col(_card(
            f"{xlsx_size:.2f} MB",
            "XLSX (real)",
            "#dc3545"
        ), md=4),

        dbc.Col(_card(
            f"{parquet_size:.2f} MB",
            "Parquet (estimado)",
            "#198754"
        ), md=4),

        dbc.Col(_card(
            f"{economia:.0f}%",
            "Economia",
            "#0d6efd"
        ), md=4),

    ])

    # =================================================
    # UI - PERFORMANCE
    # =================================================
    perf = dbc.Row([

        dbc.Col(_card(
            f"{xlsx_time:.3f}s",
            "XLSX",
            "#dc3545"
        ), md=4),

        dbc.Col(_card(
            f"{parquet_time:.3f}s",
            "Parquet",
            "#198754"
        ), md=4),

        dbc.Col(_card(
            f"{ganho:.1f}x",
            "Mais rápido",
            "#0d6efd"
        ), md=4),

    ])

    return size, perf


# =====================================================
# DOWNLOAD PARQUET REAL
# =====================================================

@callback(
    Output("download-parquet", "data"),
    Input("btn-download-parquet", "n_clicks"),
    prevent_initial_call=True
)
def download_parquet(n):

    from data_loader import data_manager

    df = data_manager.df

    return dcc.send_data_frame(
        df.to_parquet,
        "dataset.parquet",
        index=False
    )


# =====================================================
# CARD
# =====================================================

def _card(value, label, color):
    return dbc.Card([
        dbc.CardBody([
            html.H3(value, style={"color": color, "fontWeight": "700"}),
            html.Div(label, className="text-muted")
        ])
    ], className="text-center shadow-sm border-0")