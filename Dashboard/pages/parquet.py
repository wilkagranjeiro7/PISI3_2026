# pages/parquet.py
from dash import html, Input, Output, callback, dcc
import io
import time
import pandas as pd
import dash_bootstrap_components as dbc
from data_loader import data_manager


# ==================================================
# CORES PADRÃO (via DataManager)
# ==================================================

CORES = data_manager.get_cores()


def executar_benchmark(df):
    """Mede tamanho serializado e tempo de leitura nos dois formatos."""
    if df is None or df.empty:
        raise ValueError("Dataset indisponível para o benchmark.")

    csv_payload = df.to_csv(index=False)
    parquet_buffer = io.BytesIO()
    df.to_parquet(parquet_buffer, index=False, compression='snappy')
    parquet_payload = parquet_buffer.getvalue()

    inicio = time.perf_counter()
    pd.read_csv(io.StringIO(csv_payload))
    csv_time_ms = (time.perf_counter() - inicio) * 1000

    inicio = time.perf_counter()
    pd.read_parquet(io.BytesIO(parquet_payload))
    parquet_time_ms = (time.perf_counter() - inicio) * 1000

    csv_size_mb = len(csv_payload.encode('utf-8')) / 1024**2
    parquet_size_mb = len(parquet_payload) / 1024**2
    economia = ((csv_size_mb - parquet_size_mb) / csv_size_mb * 100) if csv_size_mb else 0
    speedup = (csv_time_ms / parquet_time_ms) if parquet_time_ms else 0

    return {
        'csv_size': csv_size_mb,
        'parquet_size': parquet_size_mb,
        'economia': economia,
        'csv_time': csv_time_ms,
        'parquet_time': parquet_time_ms,
        'speedup': speedup,
    }


def create_layout(df):
    """Página sobre otimização com Parquet"""

    return html.Div([

        # Botão voltar
        dbc.Button(
            "← Voltar",
            href="/",
            color="dark",
            className="mb-4",
            style={"backgroundColor": "transparent", "border": f"1px solid {CORES['border']}", "color": CORES['text']}
        ),

        # Título
        html.H1(
            "Otimização com Parquet",
            style={
                "color": CORES['text'],
                "marginBottom": "10px",
                "textAlign": "center",
                "fontSize": "36px",
                "fontWeight": "bold"
            }
        ),

        html.P(
            "Compare tamanho e tempo de leitura medidos no dataset atual",
            style={"color": CORES['text_secondary'], "textAlign": "center", "marginBottom": "40px"}
        ),

        # Botões de ação
        html.Div([

            dbc.Button(
                "Executar Benchmark",
                id="run-benchmark",
                color="success",
                className="me-2",
                style={"backgroundColor": CORES['success'], "border": "none", "color": CORES['background']}
            ),

            dbc.Button(
                "Baixar Benchmark",
                id="download-benchmark-btn",
                color="info",
                style={"backgroundColor": CORES['text_secondary'], "border": "none", "color": CORES['background']}
            ),

            dcc.Download(id="download-benchmark")

        ], className="mb-4", style={"display": "flex", "justifyContent": "center"}),

        # Cards
        html.Div([

            # Comparação de tamanho
            html.Div([

                html.H4("Comparação de Tamanho", style={"color": CORES['text'], "marginBottom": "20px"}),

                html.Div(id='parquet-tamanho')

            ],
            style={
                "backgroundColor": CORES['card_bg'],
                "color": CORES['text'],
                "border": f"1px solid {CORES['border']}",
                "borderRadius": "15px",
                "padding": "20px",
                "marginBottom": "30px"
            }),

            # Performance
            html.Div([

                html.H4("Performance de Leitura", style={"color": CORES['text'], "marginBottom": "20px"}),

                html.Div(id='parquet-performance')

            ],
            style={
                "backgroundColor": CORES['card_bg'],
                "color": CORES['text'],
                "border": f"1px solid {CORES['border']}",
                "borderRadius": "15px",
                "padding": "20px",
                "marginBottom": "30px"
            }),

            # Guia rápido
            html.Div([

                html.H4("Guia Rápido de Otimização", style={"color": CORES['text'], "marginBottom": "20px"}),

                html.Ul([

                    html.Li("Use Parquet para arquivos grandes (>100MB)", style={"color": CORES['text_secondary']}),
                    html.Li("O ganho real depende do conteúdo e do ambiente", style={"color": CORES['text_secondary']}),
                    html.Li("Preserva tipos de dados e esquema", style={"color": CORES['text_secondary']}),
                    html.Li("Ideal para dashboards e análises repetitivas", style={"color": CORES['text_secondary']}),
                    html.Li("Suporta compressão: snappy, gzip, brotli", style={"color": CORES['text_secondary']})

                ],
                style={'lineHeight': '1.8'})

            ],
            style={
                "backgroundColor": CORES['card_bg'],
                "color": CORES['text'],
                "border": f"1px solid {CORES['border']}",
                "borderRadius": "15px",
                "padding": "20px"
            })

        ])

    ],
    style={
        "backgroundColor": CORES['background'],
        "color": CORES['text'],
        "minHeight": "100vh",
        "padding": "30px"
    })


# Benchmark visual
@callback(
    Output('parquet-tamanho', 'children'),
    Output('parquet-performance', 'children'),
    Input('run-benchmark', 'n_clicks')
)
def update_parquet_info(n_clicks):

    if not n_clicks:
        return (
            html.P("Clique em Executar Benchmark", style={"color": CORES['text_secondary'], "textAlign": "center", "padding": "20px"}),
            html.P("Aguardando execução...", style={"color": CORES['text_secondary'], "textAlign": "center", "padding": "20px"})
        )

    df = data_manager.get_clean_df()
    
    if df is None:
        df = data_manager.load_data()

    try:
        benchmark = executar_benchmark(df)
    except Exception as exc:
        erro = html.P(
            f"Não foi possível executar o benchmark: {exc}",
            style={"color": CORES['danger'], "textAlign": "center"}
        )
        return erro, erro

    csv_size = benchmark['csv_size']
    parquet_size = benchmark['parquet_size']
    economia = benchmark['economia']
    csv_time = benchmark['csv_time']
    parquet_time = benchmark['parquet_time']
    speedup = benchmark['speedup']

    tamanho = html.Div([

        html.Div([
            html.H5("CSV:", style={"color": CORES['text_secondary'], "marginBottom": "5px"}),
            html.H3(f"{csv_size:.1f} MB", style={"color": CORES['text'], "marginBottom": "15px"})
        ]),

        html.Div([
            html.H5("Parquet:", style={"color": CORES['text_secondary'], "marginBottom": "5px"}),
            html.H3(f"{parquet_size:.1f} MB", style={"color": CORES['text'], "marginBottom": "15px"})
        ]),

        html.Div([
            html.H5("Economia:", style={"color": CORES['text_secondary'], "marginBottom": "5px"}),
            html.H3(
                f"{economia:.1f}%",
                style={"color": CORES['success']}
            )
        ])

    ], style={"textAlign": "center"})

    performance = html.Div([

        html.Div([
            html.H5("CSV:", style={"color": CORES['text_secondary'], "marginBottom": "5px"}),
            html.H3(f"{csv_time:.1f} ms", style={"color": CORES['text'], "marginBottom": "15px"})
        ]),

        html.Div([
            html.H5("Parquet:", style={"color": CORES['text_secondary'], "marginBottom": "5px"}),
            html.H3(f"{parquet_time:.1f} ms", style={"color": CORES['text'], "marginBottom": "15px"})
        ]),

        html.Div([
            html.H5("Velocidade:", style={"color": CORES['text_secondary'], "marginBottom": "5px"}),
            html.H3(
                f"{speedup:.2f}x",
                style={"color": CORES['success']}
            )
        ])

    ], style={"textAlign": "center"})

    return tamanho, performance


# Download CSV
@callback(
    Output("download-benchmark", "data"),
    Input("download-benchmark-btn", "n_clicks"),
    prevent_initial_call=True
)
def download_benchmark(n_clicks):

    df = data_manager.get_clean_df()
    
    if df is None:
        df = data_manager.load_data()

    benchmark = executar_benchmark(df)
    csv_size = benchmark['csv_size']
    parquet_size = benchmark['parquet_size']
    csv_time = benchmark['csv_time']
    parquet_time = benchmark['parquet_time']

    benchmark_df = pd.DataFrame({
        "Metrica": [
            "CSV Size (MB)",
            "Parquet Size (MB)",
            "Economia (%)",
            "CSV Read Time (ms)",
            "Parquet Read Time (ms)",
            "Speedup"
        ],
        "Valor": [
            round(csv_size, 2),
            round(parquet_size, 2),
            round(benchmark['economia'], 2),
            round(csv_time, 2),
            round(parquet_time, 2),
            f"{benchmark['speedup']:.2f}x"
        ]
    })

    return dcc.send_data_frame(
        benchmark_df.to_csv,
        "benchmark_parquet.csv",
        index=False
    )