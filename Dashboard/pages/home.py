# pages/home.py
from dash import html
import pandas as pd
from datetime import datetime


# ==================================================
# CORES PADRÃO DO DASHBOARD
# ==================================================

COLORS = {
    'background': '#0D0D0D',
    'card_bg': '#1A1A1A',
    'border': '#2A2A2A',
    'text': '#FFFFFF',
    'text_secondary': '#888888',
    'accent': '#3B82F6',
    'success': '#10B981',
    'warning': '#F59E0B',
    'danger': '#EF4444',
}


def create_layout(df):
    """Página Home simplificada"""

    # ================== INFO DATASET ==================
    total_registros = len(df)
    total_colunas = len(df.columns)
    total_usuarios = df['user_id'].nunique() if 'user_id' in df.columns else 0

    data_min = None
    data_max = None

    if 'date' in df.columns:
        datas = pd.to_datetime(df['date'], errors='coerce')
        data_min = datas.min()
        data_max = datas.max()

    # ================== DATA ATUAL ==================
    data_atual = datetime.now().strftime("%d/%m/%Y")

    # ================== HEADER ==================
    header = html.Div([
        html.H1(
            "Dashboard FitMatch",
            style={'textAlign': 'center',
                   'color': COLORS['text'], 'marginBottom': '10px', 'fontSize': '36px'}
        ),
        html.P(
            "Esse é um dashboard com dados de saúde para apoio ao aplicativo FitMatch",
            style={'textAlign': 'center',
                   'color': COLORS['text_secondary'], 'marginBottom': '30px'}
        ),
    ])

    # ================== INFO DATASET CARD (tamanho igual aos cards de navegação) ==================
    info_card = html.Div([
        html.Div([
            html.H4("Dataset", style={
                    'color': COLORS['text'], 'marginBottom': '15px', 'textAlign': 'center'}),
            html.Div([
                html.Div([
                    html.H3(f"{total_registros:,}", style={
                            'color': COLORS['accent'], 'marginBottom': '5px'}),
                    html.P("Registros", style={
                           'color': COLORS['text_secondary']})
                ], style={'flex': '1', 'textAlign': 'center'}),
                html.Div([
                    html.H3(f"{total_usuarios:,}", style={
                            'color': COLORS['success'], 'marginBottom': '5px'}),
                    html.P("Usuários", style={
                           'color': COLORS['text_secondary']})
                ], style={'flex': '1', 'textAlign': 'center'}),
                html.Div([
                    html.H3(f"{total_colunas}", style={
                            'color': COLORS['warning'], 'marginBottom': '5px'}),
                    html.P("Colunas", style={
                           'color': COLORS['text_secondary']})
                ], style={'flex': '1', 'textAlign': 'center'}),
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '15px'}),
            html.P(
                f"Período: {data_min.strftime('%d/%m/%Y') if pd.notnull(data_min) else 'N/A'} → {data_max.strftime('%d/%m/%Y') if pd.notnull(data_max) else 'N/A'}",
                style={'color': COLORS['text_secondary'],
                       'textAlign': 'center', 'marginTop': '10px'}
            ),
        ], style={'padding': '20px'})
    ], style={
        "backgroundColor": COLORS['card_bg'],
        "borderRadius": "10px",
        "border": f"1px solid {COLORS['border']}",
        "marginBottom": "30px",
        "flex": "1",
        "minWidth": "200px"
    })

    # ================== FUNÇÃO CARD ==================
    def card(titulo, link, descricao, icone=None):
        return html.Div([
            html.Div([
                html.Div([
                    html.I(className=f"fas fa-{icone}", style={
                           'color': COLORS['accent'], 'fontSize': '24px', 'marginRight': '10px'}) if icone else None,
                    html.H5(titulo, style={
                            'color': COLORS['text'], 'marginBottom': '10px', 'fontWeight': '600'}),
                ], style={'display': 'flex', 'alignItems': 'center'}),
                html.P(descricao, style={
                       'color': COLORS['text_secondary'], 'marginBottom': '15px', 'fontSize': '14px'}),
                html.A(
                    "Acessar →",
                    href=link,
                    className="btn",
                    style={
                        'backgroundColor': 'transparent',
                        'border': f'1px solid {COLORS["border"]}',
                        'color': COLORS['text'],
                        'borderRadius': '4px',
                        'padding': '5px 15px',
                        'fontSize': '12px',
                        'textDecoration': 'none',
                        'display': 'inline-block',
                        'transition': 'all 0.3s ease'
                    }
                )
            ])
        ], style={
            "backgroundColor": COLORS['card_bg'],
            "padding": "20px",
            "borderRadius": "10px",
            "border": f"1px solid {COLORS['border']}",
            "flex": "1",
            "minWidth": "200px",
            "transition": "all 0.3s ease"
        })

    # ================== LINHA DO DATASET CARD (sozinho para destacar) ==================
    dataset_row = html.Div([
        info_card
    ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '30px'})

    # ================== NAVEGAÇÃO (3 linhas x 3 cards + 1 linha extra para Insights) ==================
    navigation = html.Div([

        # LINHA 1
        html.Div([
            card("Dataframes", "/dataframes",
                 "Exploração completa dos dados", "table"),
            card("EDA", "/eda", "Análise exploratória interativa", "chart-line"),
            card("Profiling", "/profiling",
                 "Resumo da qualidade dos dados", "clipboard-list"),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px', 'flexWrap': 'wrap'}),

        # LINHA 2
        html.Div([
            card("Plots", "/plots", "Visualizações interativas", "chart-bar"),
            card("Parquet", "/parquet", "Otimização com Parquet", "file-archive"),
            card("Filtros", "/filtros", "Filtragem interativa", "filter"),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px', 'flexWrap': 'wrap'}),

        # LINHA 3
        html.Div([
            card("Agrupamentos", "/agrupamentos",
                 "Agregações e group by", "layer-group"),
            card("K-Means", "/kmeans", "Clusterização de usuários", "circle-nodes"),
            card("Classificação", "/classificacao",
                 "Modelos de machine learning", "robot"),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px', 'flexWrap': 'wrap'}),

        # LINHA 4 - INSIGHTS (DESTAQUE)
        html.Div([
            card(
                "Insights",
                "/insights",
                "Análise de recuperação e recomendações personalizadas",
                "lightbulb"
            ),
            card(
                "Classificação Avançada",
                "/advanced-classification",
                "Pipeline com XAI e Balanceamento",
                "brain"
            ),
            card(
                "Pipeline Leonardo",
                "/pipeline-leonardo",
                "V.A Final - Novo modelo de classificação e pipeline",
                "chart-column"
            ),



        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px', 'flexWrap': 'wrap'}),
        html.Div([
            card(
                "Repositorios de Graficos Plotados - Leaonardo",
                "/pipeline-leonardo-resultado",
                "   Repositórios dos Gráficos - 7 arquivos disponíveis",
                "chart-column"
            ),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px', 'flexWrap': 'wrap'}),
    ])

    # ================== RODAPÉ ==================
    footer = html.Div([
        html.P(
            f"Recife | {data_atual}",
            style={'textAlign': 'center', 'color': COLORS['text_secondary'], 'fontSize': '12px',
                   'marginTop': '30px', 'paddingTop': '20px', 'borderTop': f'1px solid {COLORS["border"]}'}
        )
    ])

    # ================== LAYOUT FINAL ==================
    return html.Div([
        header,
        dataset_row,
        navigation,
        footer
    ], style={
        "backgroundColor": COLORS['background'],
        "minHeight": "100vh",
        "padding": "30px",
        "fontFamily": "'Segoe UI', 'Roboto', sans-serif"
    })
