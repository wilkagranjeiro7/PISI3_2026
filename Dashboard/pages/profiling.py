# pages/profiling.py

from dash import html, dcc, Input, Output, callback
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
from data_loader import data_manager


# ==================================================
# CORES PADRÃO (via DataManager)
# ==================================================

CORES = data_manager.get_cores()
TEMPLATE = 'plotly_dark'
ALTURA_PADRAO = 450


# ==================================================
# OBTEM DADOS
# ==================================================

def get_df():
    """Obtém o DataFrame tratado do DataManager"""
    df = data_manager.get_clean_df()
    
    if df is None:
        df = data_manager.load_data()
    
    if df is None:
        return pd.DataFrame()
    
    df = df.copy()
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    
    return df


# ==================================================
# LAYOUT
# ==================================================

def create_layout(df):
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
            "Data Profiling",
            style={
                "color": CORES['text'],
                "marginBottom": "10px",
                "textAlign": "center",
                "fontSize": "36px",
                "fontWeight": "bold"
            }
        ),
        
        html.P(
            "Análise da qualidade e estrutura dos dados",
            style={"color": CORES['text_secondary'], "textAlign": "center", "marginBottom": "40px"}
        ),
        
        # GRÁFICO 1 - Tipos de Dados
        html.Div([
            html.H4("Tipos de Dados", style={"color": CORES['text'], "marginBottom": "20px"}),
            dcc.Graph(id="profiling-dtypes", config={'displayModeBar': False})
        ], style={"backgroundColor": CORES['card_bg'], "padding": "20px", "borderRadius": "15px", "marginBottom": "30px", "border": f"1px solid {CORES['border']}"}),
        
        # GRÁFICO 2 - Valores Nulos
        html.Div([
            html.H4("Valores Nulos por Coluna", style={"color": CORES['text'], "marginBottom": "20px"}),
            dcc.Graph(id="profiling-nulos", config={'displayModeBar': False})
        ], style={"backgroundColor": CORES['card_bg'], "padding": "20px", "borderRadius": "15px", "marginBottom": "30px", "border": f"1px solid {CORES['border']}"}),
        
        # GRÁFICO 3 - Outliers
        html.Div([
            html.H4("Outliers Detectados", style={"color": CORES['text'], "marginBottom": "20px"}),
            dcc.Graph(id="profiling-outliers", config={'displayModeBar': False})
        ], style={"backgroundColor": CORES['card_bg'], "padding": "20px", "borderRadius": "15px", "marginBottom": "30px", "border": f"1px solid {CORES['border']}"}),
        
    ], style={
        "backgroundColor": CORES['background'],
        "minHeight": "100vh",
        "padding": "30px"
    })


# ==================================================
# TIPOS DE DADOS
# ==================================================

@callback(
    Output("profiling-dtypes", "figure"),
    Input("profiling-dtypes", "id")
)
def update_dtypes(_):
    df = get_df()
    
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Sem dados disponíveis",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color=CORES['text_secondary'], size=18)
        )
        fig.update_layout(
            template=TEMPLATE,
            height=ALTURA_PADRAO,
            paper_bgcolor=CORES['card_bg'],
            plot_bgcolor=CORES['card_bg']
        )
        return fig
    
    # Contar tipos de dados
    dtypes = df.dtypes.astype(str).value_counts().reset_index()
    dtypes.columns = ['tipo', 'quantidade']
    
    # Mapear nomes amigáveis
    type_names = {
        'object': 'Texto/Objeto',
        'int64': 'Inteiro',
        'float64': 'Decimal',
        'datetime64[ns]': 'Data/Hora',
        'bool': 'Booleano'
    }
    
    dtypes['tipo_amigavel'] = dtypes['tipo'].map(lambda x: type_names.get(x, x))
    dtypes = dtypes.sort_values('quantidade', ascending=True)
    
    # Gráfico de barras horizontais
    fig = go.Figure(data=[
        go.Bar(
            y=dtypes['tipo_amigavel'],
            x=dtypes['quantidade'],
            orientation='h',
            marker=dict(
                color=CORES['chart_colors'][:len(dtypes)],
                cornerradius=5,
                line=dict(color=CORES['card_bg'], width=0.5)
            ),
            text=dtypes['quantidade'],
            textposition='outside',
            textfont=dict(color=CORES['text'], size=12),
            hovertemplate='<b>%{y}</b><br>Quantidade: %{x}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        template=TEMPLATE,
        height=ALTURA_PADRAO,
        paper_bgcolor=CORES['card_bg'],
        plot_bgcolor=CORES['card_bg'],
        font_color=CORES['text'],
        xaxis=dict(
            title="Quantidade de Colunas",
            gridcolor=CORES['border'],
            tickfont=dict(size=11)
        ),
        yaxis=dict(title="", gridcolor=CORES['border']),
        margin=dict(t=20, b=20, l=120, r=40),
        showlegend=False
    )
    
    return fig


# ==================================================
# VALORES NULOS POR COLUNA
# ==================================================

@callback(
    Output("profiling-nulos", "figure"),
    Input("profiling-nulos", "id")
)
def update_nulos(_):
    df = get_df()
    
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Sem dados disponíveis",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color=CORES['text_secondary'], size=18)
        )
        fig.update_layout(
            template=TEMPLATE,
            height=ALTURA_PADRAO,
            paper_bgcolor=CORES['card_bg'],
            plot_bgcolor=CORES['card_bg']
        )
        return fig
    
    # Calcular percentual de nulos por coluna
    nulos = (df.isnull().sum() / len(df)) * 100
    nulos = nulos.sort_values(ascending=True)
    
    # Filtrar apenas colunas com pelo menos 1% de nulos
    nulos_com_problema = nulos[nulos > 1]
    
    if nulos_com_problema.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Nenhuma coluna com valores nulos significativos!",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color=CORES['success'], size=18)
        )
        fig.update_layout(
            template=TEMPLATE,
            height=ALTURA_PADRAO,
            paper_bgcolor=CORES['card_bg'],
            plot_bgcolor=CORES['card_bg']
        )
        return fig
    
    # Cores baseadas no percentual
    cores = [
        CORES['success'] if x < 10 
        else CORES['warning'] if x < 30 
        else CORES['danger'] 
        for x in nulos_com_problema.values
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            y=nulos_com_problema.index,
            x=nulos_com_problema.values,
            orientation='h',
            marker=dict(color=cores, cornerradius=5),
            text=[f'{x:.1f}%' for x in nulos_com_problema.values],
            textposition='outside',
            textfont=dict(color=CORES['text'], size=10),
            hovertemplate='<b>%{y}</b><br>Nulos: %{x:.1f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        template=TEMPLATE,
        height=ALTURA_PADRAO,
        paper_bgcolor=CORES['card_bg'],
        plot_bgcolor=CORES['card_bg'],
        font_color=CORES['text'],
        xaxis=dict(
            title="Percentual de Valores Nulos (%)",
            gridcolor=CORES['border'],
            range=[0, 100],
            tickvals=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        ),
        yaxis=dict(title="", gridcolor=CORES['border']),
        margin=dict(t=20, b=20, l=150, r=40),
        showlegend=False
    )
    
    return fig


# ==================================================
# OUTLIERS - COM EIXOS INVERTIDOS (HORIZONTAL)
# ==================================================

@callback(
    Output("profiling-outliers", "figure"),
    Input("profiling-outliers", "id")
)
def update_outliers(_):
    # Usar o método público get_outliers() do DataManager
    outliers_dict = data_manager.get_outliers()
    
    tem_outliers = False
    dados_outliers = []  # Lista para armazenar (nome, valores)
    
    if outliers_dict:
        for categoria, valores in outliers_dict.items():
            if isinstance(valores, dict):
                for campo, info in valores.items():
                    if info.get('count', 0) > 0:
                        tem_outliers = True
                        nome = f"{categoria} - {campo}"
                        dados_outliers.append({
                            'nome': nome,
                            'valores': info.get('values', [])
                        })
    
    # Se não há outliers
    if not tem_outliers:
        fig = go.Figure()
        fig.add_annotation(
            text="Nenhum outlier encontrado nos dados!",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color=CORES['success'], size=18)
        )
        fig.update_layout(
            template=TEMPLATE,
            height=ALTURA_PADRAO,
            paper_bgcolor=CORES['card_bg'],
            plot_bgcolor=CORES['card_bg']
        )
        return fig
    
    # Criar gráfico de boxplot horizontal
    fig = go.Figure()
    
    for i, item in enumerate(dados_outliers):
        if len(item['valores']) > 0:
            fig.add_trace(go.Box(
                y=[item['nome']] * len(item['valores']),  # Nome repetido para cada valor
                x=item['valores'],  # Valores no eixo X
                name=item['nome'],
                orientation='h',  # Horizontal
                marker_color=CORES['warning'],
                boxmean='sd',
                hovertemplate='<b>%{y}</b><br>Valor: %{x}<extra></extra>'
            ))
    
    # Configurar layout do gráfico
    fig.update_layout(
        template=TEMPLATE,
        height=ALTURA_PADRAO,
        paper_bgcolor=CORES['card_bg'],
        plot_bgcolor=CORES['card_bg'],
        font_color=CORES['text'],
        xaxis=dict(title="Valores dos Outliers", gridcolor=CORES['border']),
        yaxis=dict(title="", gridcolor=CORES['border']),
        showlegend=False,
        margin=dict(t=40, b=40, l=150, r=40),  # Espaço para os nomes
        boxgap=0.3,
        boxgroupgap=0.1
    )
    
    return fig