# pages/agrupamentos.py
import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, callback
import plotly.express as px
import pandas as pd
from data_loader import data_manager


# ==================================================
# CORES PADRÃO (via DataManager)
# ==================================================

CORES = data_manager.get_cores()
TEMPLATE = 'plotly_dark'


def create_layout(df=None):
    """Página de agrupamentos no padrão do dashboard"""

    # Opções para os selects
    group_options = [
        {"label": data_manager.traduzir_coluna(col), "value": col}
        for col in ["fitness_level", "gender", "primary_sport", "activity_type", "workout_time_of_day"]
        if col in df.columns
    ]

    metric_options = [
        {"label": data_manager.traduzir_coluna(col), "value": col}
        for col in ["recovery_score", "day_strain", "sleep_hours", "hrv", "calories_burned"]
        if col in df.columns
    ]

    agg_options = [
        {"label": "Média", "value": "mean"},
        {"label": "Mediana", "value": "median"},
        {"label": "Máximo", "value": "max"},
        {"label": "Mínimo", "value": "min"},
        {"label": "Desvio Padrão", "value": "std"}
    ]

    # Estilo padrão para os selects
    SELECT_STYLE = {
        'backgroundColor': CORES['card_bg'],
        'color': CORES['text'],
        'border': f'1px solid {CORES["border"]}',
        'borderRadius': '4px'
    }

    return html.Div([
        # Botão voltar (fixo no canto superior esquerdo)
        html.Div([
            dbc.Button(
                "← Voltar",
                href="/",
                color="light",
                size="sm",
                style={'backgroundColor': 'transparent', 'border': f'1px solid {CORES["border"]}', 
                       'color': CORES['text']}
            )
        ], style={'position': 'fixed', 'top': '20px', 'left': '20px', 'zIndex': '1000'}),

        # Conteúdo principal
        html.Div([
            # Painel esquerdo - configurações (FIXO)
            html.Div([
                html.H3("Agrupamentos", style={'fontWeight': 'normal', 'marginBottom': '30px', 'color': CORES['text']}),

                # Agrupar por
                html.Div([
                    html.Label("AGRUPAR POR", style={'color': CORES['text_secondary'], 'fontSize': '12px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                    html.Div([
                        dbc.Select(
                            id='groupby-col',
                            options=group_options,
                            value=group_options[0]['value'] if group_options else None,
                            style=SELECT_STYLE
                        )
                    ], style={'marginTop': '10px'})
                ], style={'marginBottom': '30px'}),

                # Métrica para agregar
                html.Div([
                    html.Label("MÉTRICA", style={'color': CORES['text_secondary'], 'fontSize': '12px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                    html.Div([
                        dbc.Select(
                            id='agg-metric',
                            options=metric_options,
                            value=metric_options[0]['value'] if metric_options else None,
                            style=SELECT_STYLE
                        )
                    ], style={'marginTop': '10px'})
                ], style={'marginBottom': '30px'}),

                # Função de Agregação
                html.Div([
                    html.Label("FUNÇÃO", style={'color': CORES['text_secondary'], 'fontSize': '12px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                    html.Div([
                        dbc.Select(
                            id='agg-func',
                            options=agg_options,
                            value='mean',
                            style=SELECT_STYLE
                        )
                    ], style={'marginTop': '10px'})
                ], style={'marginBottom': '30px'}),

            ], style={
                'position': 'fixed', 
                'width': '300px', 
                'padding': '80px 25px 20px 25px',
                'borderRight': f'1px solid {CORES["border"]}',
                'height': '100vh',
                'overflowY': 'auto',
                'backgroundColor': CORES['background']
            }),

            # Painel direito - gráfico apenas (sem tabela)
            html.Div([
                html.Div(id='groupby-container', children=[
                    html.Div([
                        html.P("Selecione as opções para visualizar o gráfico", 
                              style={'color': CORES['text_secondary'], 'textAlign': 'center', 'marginTop': '50px'})
                    ])
                ])
            ], style={'marginLeft': '320px', 'padding': '20px', 'minHeight': '100vh'})

        ])

    ], style={'backgroundColor': CORES['background'], 'minHeight': '100vh', 'color': CORES['text']})


# ==================================================
# CALLBACK
# ==================================================

@callback(
    Output('groupby-container', 'children'),
    Input('groupby-col', 'value'),
    Input('agg-metric', 'value'),
    Input('agg-func', 'value')
)
def update_groupby(group_col, metric, agg_func):
    """Executa o agrupamento e retorna apenas o gráfico"""

    # Usar o método correto do DataManager
    df = data_manager.get_clean_df()

    if df is None or df.empty:
        return html.Div("❌ Dados não disponíveis", style={'color': CORES['danger'], 'textAlign': 'center', 'padding': 50})

    if group_col is None or metric is None:
        return html.Div("⚠️ Selecione as opções para visualizar os resultados", 
                        style={'color': CORES['warning'], 'textAlign': 'center', 'padding': 50})

    # Verificar se as colunas existem
    if group_col not in df.columns:
        return html.Div(f"❌ Coluna '{group_col}' não encontrada", style={'color': CORES['danger'], 'textAlign': 'center', 'padding': 50})

    if metric not in df.columns:
        return html.Div(f"❌ Coluna '{metric}' não encontrada", style={'color': CORES['danger'], 'textAlign': 'center', 'padding': 50})

    # Remover valores nulos
    df_clean = df[[group_col, metric]].dropna()

    if df_clean.empty:
        return html.Div("⚠️ Sem dados suficientes após remoção de valores nulos", 
                        style={'color': CORES['warning'], 'textAlign': 'center', 'padding': 50})

    # ===================================
    # GROUPBY
    # ===================================
    grouped = df_clean.groupby(group_col)[metric].agg(agg_func).reset_index()
    grouped = grouped.sort_values(metric, ascending=False)

    # ===================================
    # NOMES TRADUZIDOS
    # ===================================
    func_names = {
        'mean': 'Média',
        'median': 'Mediana',
        'max': 'Máximo',
        'min': 'Mínimo',
        'std': 'Desvio Padrão'
    }

    col_nome = data_manager.traduzir_coluna(group_col)
    metric_nome = data_manager.traduzir_coluna(metric)
    func_nome = func_names.get(agg_func, agg_func.upper())

    # ===================================
    # GRÁFICO COM EIXOS INVERTIDOS (HORIZONTAL)
    # ===================================
    grafico_df = grouped.copy()
    grafico_df[group_col] = grafico_df[group_col].apply(
        lambda x: data_manager.VALUE_TRANSLATIONS.get(str(x).lower(), x) if pd.notna(x) else x
    )
    
    # Ordenar para o gráfico horizontal (melhor visualização)
    grafico_df = grafico_df.sort_values(metric, ascending=True)

    # Escala de cores baseada na métrica
    if metric == 'recovery_score':
        color_scale = 'Greens'
    elif metric == 'sleep_hours':
        color_scale = 'Blues'
    elif metric == 'hrv':
        color_scale = 'Purples'
    elif metric == 'day_strain':
        color_scale = 'Oranges'
    else:
        color_scale = 'Viridis'

    fig = px.bar(
        grafico_df,
        y=group_col,  # Eixo Y recebe a categoria
        x=metric,     # Eixo X recebe o valor
        color=metric,
        color_continuous_scale=color_scale,
        title=f"{func_nome} de {metric_nome} por {col_nome}",
        text=metric,
        orientation='h'  # Barras horizontais
    )

    fig.update_traces(
        texttemplate='%{text:.2f}', 
        textposition='outside',
        textfont=dict(size=11)
    )
    
    # Altura dinâmica baseada na quantidade de itens
    altura = max(500, len(grafico_df) * 40)
    
    fig.update_layout(
        template=TEMPLATE,
        paper_bgcolor=CORES['card_bg'],
        plot_bgcolor=CORES['card_bg'],
        font_color=CORES['text'],
        title_font_color=CORES['text'],
        title_x=0.5,
        height=altura,
        xaxis=dict(gridcolor=CORES['border'], title=f"{func_nome} de {metric_nome}"),
        yaxis=dict(gridcolor=CORES['border'], title=col_nome, tickfont=dict(size=11)),
        coloraxis_colorbar=dict(title=metric_nome)
    )

    grafico = dcc.Graph(figure=fig, config={'displayModeBar': True, 'displaylogo': False})

    return html.Div([grafico])