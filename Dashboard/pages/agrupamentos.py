# pages/agrupamentos.py
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, callback
import plotly.express as px
import pandas as pd

def create_layout(df):
    """Página para análises de agrupamento"""
    
    return html.Div([
        html.H1("📈 Análises de Agrupamento (GroupBy)", className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Configurações"),
                    dbc.CardBody([
                        html.Label("Agrupar por:"),
                        dcc.Dropdown(
                            id='groupby-col',
                            options=[
                                {'label': 'Nível de Fitness', 'value': 'fitness_level'},
                                {'label': 'Gênero', 'value': 'gender'},
                                {'label': 'Dia da Semana', 'value': 'day_of_week'},
                                {'label': 'Esporte Principal', 'value': 'primary_sport'},
                                {'label': 'Tipo de Atividade', 'value': 'activity_type'}
                            ],
                            value='fitness_level'
                        ),
                        html.Br(),
                        
                        html.Label("Métrica para agregar:"),
                        dcc.Dropdown(
                            id='agg-metric',
                            options=[
                                {'label': 'Recovery Score', 'value': 'recovery_score'},
                                {'label': 'Day Strain', 'value': 'day_strain'},
                                {'label': 'Horas de Sono', 'value': 'sleep_hours'},
                                {'label': 'HRV', 'value': 'hrv'},
                                {'label': 'Calorias', 'value': 'calories_burned'}
                            ],
                            value='recovery_score'
                        ),
                        html.Br(),
                        
                        html.Label("Função de Agregação:"),
                        dcc.Dropdown(
                            id='agg-func',
                            options=[
                                {'label': 'Média', 'value': 'mean'},
                                {'label': 'Mediana', 'value': 'median'},
                                {'label': 'Máximo', 'value': 'max'},
                                {'label': 'Mínimo', 'value': 'min'},
                                {'label': 'Desvio Padrão', 'value': 'std'}
                            ],
                            value='mean'
                        )
                    ])
                ])
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Resultados do Agrupamento"),
                    dbc.CardBody([
                        html.Div(id='groupby-results'),
                        dcc.Graph(id='groupby-graph')
                    ])
                ])
            ], width=9)
        ])
    ])

@callback(
    [Output('groupby-results', 'children'),
     Output('groupby-graph', 'figure')],
    [Input('groupby-col', 'value'),
     Input('agg-metric', 'value'),
     Input('agg-func', 'value')]
)
def update_groupby(group_col, metric, agg_func):
    from data_loader import data_manager
    df = data_manager.df
    
    # Realizar agrupamento
    grouped = df.groupby(group_col)[metric].agg(agg_func).reset_index()
    grouped = grouped.sort_values(metric, ascending=False)
    
    # Tabela de resultados
    table = dbc.Table.from_dataframe(grouped.head(20), striped=True, bordered=True)
    
    # Gráfico
    fig = px.bar(grouped, x=group_col, y=metric, 
                 title=f'{agg_func.upper()} de {metric} por {group_col}',
                 color=metric, color_continuous_scale='Viridis')
    fig.update_layout(xaxis_title=group_col, yaxis_title=f'{agg_func} de {metric}',
                     template='plotly_white')
    
    return table, fig