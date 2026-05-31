# pages/profiling.py
from dash import html, dcc, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_layout(df):
    """Página de profiling - análise de qualidade dos dados"""
    
    return html.Div([
        html.H1("🔬 Data Profiling - Qualidade dos Dados", style={'marginBottom': 20}),
        
        html.Div([
            html.Div([
                html.H4("📊 Resumo Geral", style={'marginBottom': 15}),
                html.Div([
                    html.Div([html.Div(f"{len(df):,}", className="stat-number"), html.Div("Total Registros")], className="stat-card"),
                    html.Div([html.Div(f"{df['user_id'].nunique()}", className="stat-number"), html.Div("Usuários")], className="stat-card"),
                    html.Div([html.Div(f"{len(df.columns)}", className="stat-number"), html.Div("Colunas")], className="stat-card"),
                    html.Div([html.Div(f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB", className="stat-number"), html.Div("Memória")], className="stat-card"),
                ], className="stats-grid"),
                html.Div([
                    html.P(f"📅 Período: {df['date'].min().date()} a {df['date'].max().date()}"),
                    html.P(f"❌ Valores nulos totais: {df.isna().sum().sum()}"),
                    html.P(f"✅ Colunas completas (0 nulos): {(df.isna().sum() == 0).sum()} de {len(df.columns)}"),
                ])
            ], className="chart-card"),
            
            html.Div([
                html.H4("📊 Distribuição de Tipos de Dados", className="chart-title"),
                dcc.Graph(id='profiling-dtypes')
            ], className="chart-card"),
            
            html.Div([
                html.H4("❌ Top 10 Colunas com Mais Valores Nulos", className="chart-title"),
                dcc.Graph(id='profiling-nulos')
            ], className="chart-card"),
            
            html.Div([
                html.H4("📈 Distribuição das Principais Métricas", className="chart-title"),
                dcc.Graph(id='profiling-distribuicoes')
            ], className="chart-card"),
        ])
    ])

@callback(
    Output('profiling-dtypes', 'figure'),
    Input('profiling-dtypes', 'id')
)
def update_dtypes(_):
    from data_loader import data_manager
    df = data_manager.df
    
    dtypes = df.dtypes.astype(str).value_counts()
    fig = px.pie(values=dtypes.values, names=dtypes.index, title='Distribuição de Tipos de Dados')
    fig.update_layout(template='plotly_white', height=400)
    return fig

@callback(
    Output('profiling-nulos', 'figure'),
    Input('profiling-nulos', 'id')
)
def update_nulos(_):
    from data_loader import data_manager
    df = data_manager.df
    
    nulos = df.isna().sum()
    nulos = nulos[nulos > 0].sort_values(ascending=False).head(10)
    
    if len(nulos) > 0:
        fig = px.bar(x=nulos.values, y=nulos.index, orientation='h',
                    # pages/profiling.py (continuação)
                     title='Colunas com Valores Nulos',
                     labels={'x': 'Quantidade de Nulos', 'y': 'Coluna'},
                     color_discrete_sequence=['#e74c3c'])
        fig.update_layout(template='plotly_white', height=400)
        return fig
    else:
        fig = go.Figure()
        fig.add_annotation(text="✅ Sem valores nulos encontrados!", x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        fig.update_layout(template='plotly_white', height=400)
        return fig

@callback(
    Output('profiling-distribuicoes', 'figure'),
    Input('profiling-distribuicoes', 'id')
)
def update_distribuicoes(_):
    from data_loader import data_manager
    df = data_manager.df
    
    from plotly.subplots import make_subplots
    
    metrics = ['recovery_score', 'day_strain', 'sleep_hours', 'hrv']
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=metrics)
    
    for i, metric in enumerate(metrics):
        row = i // 2 + 1
        col = i % 2 + 1
        fig.add_trace(go.Histogram(x=df[metric], name=metric, 
                                   marker_color=colors[i], nbinsx=30), 
                     row=row, col=col)
    
    fig.update_layout(title='Distribuição das Principais Métricas',
                     template='plotly_white', height=600, showlegend=False)
    fig.update_xaxes(title_text="Valor", row=2, col=1)
    fig.update_xaxes(title_text="Valor", row=2, col=2)
    fig.update_yaxes(title_text="Frequência", row=1, col=1)
    
    return fig