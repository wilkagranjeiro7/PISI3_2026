# pages/filtros.py
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from data_loader import data_manager


# ==================================================
# CORES PADRÃO (via DataManager)
# ==================================================

CORES = data_manager.get_cores()
TEMPLATE = 'plotly_dark'
DEFAULT_COLORS = CORES['chart_colors']


def create_layout(df):
    """Página com filtros interativos no padrão do dashboard"""

    # Garantir que os valores estão traduzidos para os dropdowns
    fitness_values = sorted(df["fitness_level"].unique()) if "fitness_level" in df.columns else []
    gender_values = sorted(df["gender"].unique()) if "gender" in df.columns else []
    sport_values = sorted(df["primary_sport"].unique()) if "primary_sport" in df.columns else []

    fitness_options = [{"label": "Todos", "value": "Todos"}] + [
        {"label": x, "value": x} for x in fitness_values
    ]

    gender_options = [{"label": "Todos", "value": "Todos"}] + [
        {"label": x, "value": x} for x in gender_values
    ]

    sport_options = [{"label": "Todos", "value": "Todos"}] + [
        {"label": x, "value": x} for x in sport_values
    ]

    # Estilo padrão para os inputs
    INPUT_STYLE = {
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
            # Painel esquerdo - filtros (FIXO)
            html.Div([
                html.H3("Filtros", style={'fontWeight': 'normal', 'marginBottom': '30px', 'color': CORES['text']}),

                # Nível de Fitness
                html.Div([
                    html.Label("NÍVEL DE FITNESS", style={'color': CORES['text_secondary'], 'fontSize': '12px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                    html.Div([
                        dbc.RadioItems(
                            id="filtro-fitness",
                            options=fitness_options,
                            value="Todos",
                            className="mb-4",
                            style={'marginTop': '10px'}
                        )
                    ])
                ], style={'marginBottom': '30px'}),

                # Gênero
                html.Div([
                    html.Label("GÊNERO", style={'color': CORES['text_secondary'], 'fontSize': '12px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                    html.Div([
                        dbc.RadioItems(
                            id="filtro-genero",
                            options=gender_options,
                            value="Todos",
                            className="mb-4",
                            style={'marginTop': '10px'}
                        )
                    ])
                ], style={'marginBottom': '30px'}),

                # Esporte Principal
                html.Div([
                    html.Label("ESPORTE PRINCIPAL", style={'color': CORES['text_secondary'], 'fontSize': '12px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                    html.Div([
                        dbc.RadioItems(
                            id="filtro-esporte",
                            options=sport_options,
                            value="Todos",
                            className="mb-4",
                            style={'marginTop': '10px'}
                        )
                    ])
                ], style={'marginBottom': '30px'}),

                html.Hr(style={'borderColor': CORES['border'], 'margin': '20px 0'}),

                # Recovery Score
                html.Div([
                    html.Label("RECOVERY SCORE", style={'color': CORES['text_secondary'], 'fontSize': '12px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                    html.Div([
                        dbc.Row([
                            dbc.Col(
                                dbc.Input(
                                    id="recovery-min",
                                    type="number",
                                    value=0,
                                    min=0,
                                    max=100,
                                    placeholder="Mínimo",
                                    style=INPUT_STYLE
                                ),
                                width=6
                            ),
                            dbc.Col(
                                dbc.Input(
                                    id="recovery-max",
                                    type="number",
                                    value=100,
                                    min=0,
                                    max=100,
                                    placeholder="Máximo",
                                    style=INPUT_STYLE
                                ),
                                width=6
                            )
                        ])
                    ], style={'marginTop': '10px'})
                ], style={'marginBottom': '30px'}),

                # Horas de Sono
                html.Div([
                    html.Label("HORAS DE SONO", style={'color': CORES['text_secondary'], 'fontSize': '12px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                    html.Div([
                        dbc.Row([
                            dbc.Col(
                                dbc.Input(
                                    id="sono-min",
                                    type="number",
                                    value=0,
                                    min=0,
                                    max=24,
                                    placeholder="Mínimo",
                                    style=INPUT_STYLE
                                ),
                                width=6
                            ),
                            dbc.Col(
                                dbc.Input(
                                    id="sono-max",
                                    type="number",
                                    value=24,
                                    min=0,
                                    max=24,
                                    placeholder="Máximo",
                                    style=INPUT_STYLE
                                ),
                                width=6
                            )
                        ])
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
            
            # Painel direito - resultados
            html.Div([
                html.Div(id='filtro-resultados-container', children=[
                    html.Div([
                        html.P("Selecione os filtros para visualizar os resultados", 
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
    Output('filtro-resultados-container', 'children'),
    Input("filtro-fitness", "value"),
    Input("filtro-genero", "value"),
    Input("filtro-esporte", "value"),
    Input("recovery-min", "value"),
    Input("recovery-max", "value"),
    Input("sono-min", "value"),
    Input("sono-max", "value")
)
def aplicar_filtros(fitness, genero, esporte, recovery_min, recovery_max, sono_min, sono_max):
    """Aplica os filtros e retorna os resultados"""

    # Usar o método correto do DataManager
    df = data_manager.get_clean_df()
    
    if df is None or df.empty:
        return html.Div("❌ Dados não disponíveis", style={'color': CORES['danger'], 'textAlign': 'center', 'padding': 50})

    df = df.copy()

    # Garantir que as colunas existem
    if "fitness_level" in df.columns and fitness != "Todos":
        df = df[df["fitness_level"] == fitness]

    if "gender" in df.columns and genero != "Todos":
        df = df[df["gender"] == genero]

    if "primary_sport" in df.columns and esporte != "Todos":
        df = df[df["primary_sport"] == esporte]

    # Filtros numéricos
    if "recovery_score" in df.columns:
        recovery_min = recovery_min if recovery_min is not None else 0
        recovery_max = recovery_max if recovery_max is not None else 100
        df = df[(df["recovery_score"] >= recovery_min) & (df["recovery_score"] <= recovery_max)]

    if "sleep_hours" in df.columns:
        sono_min = sono_min if sono_min is not None else 0
        sono_max = sono_max if sono_max is not None else 24
        df = df[(df["sleep_hours"] >= sono_min) & (df["sleep_hours"] <= sono_max)]

    # Caso vazio
    if df.empty:
        return html.Div([
            html.H5("Nenhum registro encontrado.", style={'color': CORES['warning'], 'textAlign': 'center', 'padding': 50})
        ])

    # Calcular estatísticas
    total_registros = len(df)
    usuarios_unicos = df['user_id'].nunique() if 'user_id' in df.columns else 0
    
    recovery_medio = df['recovery_score'].mean() if 'recovery_score' in df.columns else 0
    sono_medio = df['sleep_hours'].mean() if 'sleep_hours' in df.columns else 0
    strain_medio = df['day_strain'].mean() if 'day_strain' in df.columns else 0
    hrv_medio = df['hrv'].mean() if 'hrv' in df.columns else 0

    # Cards de estatísticas (um único card com 6 colunas em uma linha)
    stats_cards = html.Div([
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H4(f"{total_registros:,}", className="card-title", style={'color': CORES['accent'], 'fontSize': '28px', 'marginBottom': '5px'}),
                        html.P("Registros", style={'color': CORES['text_secondary'], 'marginBottom': '0'})
                    ], md=2, className="text-center"),
                    
                    dbc.Col([
                        html.H4(f"{usuarios_unicos:,}", className="card-title", style={'color': CORES['accent'], 'fontSize': '28px', 'marginBottom': '5px'}),
                        html.P("Usuários", style={'color': CORES['text_secondary'], 'marginBottom': '0'})
                    ], md=2, className="text-center"),
                    
                    dbc.Col([
                        html.H4(f"{recovery_medio:.1f}", className="card-title", style={'color': CORES['success'], 'fontSize': '28px', 'marginBottom': '5px'}),
                        html.P("Recovery Médio", style={'color': CORES['text_secondary'], 'marginBottom': '0'})
                    ], md=2, className="text-center"),
                    
                    dbc.Col([
                        html.H4(f"{sono_medio:.1f}h", className="card-title", style={'color': CORES['sleep'], 'fontSize': '28px', 'marginBottom': '5px'}),
                        html.P("Sono Médio", style={'color': CORES['text_secondary'], 'marginBottom': '0'})
                    ], md=2, className="text-center"),
                    
                    dbc.Col([
                        html.H4(f"{strain_medio:.1f}", className="card-title", style={'color': CORES['warning'], 'fontSize': '28px', 'marginBottom': '5px'}),
                        html.P("Strain Médio", style={'color': CORES['text_secondary'], 'marginBottom': '0'})
                    ], md=2, className="text-center"),
                    
                    dbc.Col([
                        html.H4(f"{hrv_medio:.1f} ms", className="card-title", style={'color': CORES['hrv'], 'fontSize': '28px', 'marginBottom': '5px'}),
                        html.P("HRV Médio", style={'color': CORES['text_secondary'], 'marginBottom': '0'})
                    ], md=2, className="text-center"),
                ], className="g-0")
            ])
        ], style={'backgroundColor': CORES['card_bg'], 'border': f'1px solid {CORES["border"]}', 'borderRadius': '10px', 'marginBottom': '20px'})
    ])

    # Gráfico
    if 'recovery_score' in df.columns and 'day_strain' in df.columns:
        fig = px.scatter(
            df, 
            x='recovery_score', 
            y='day_strain',
            title="Recovery Score vs Day Strain",
            labels={'recovery_score': 'Recovery Score', 'day_strain': 'Day Strain'},
            color_discrete_sequence=[DEFAULT_COLORS[0]],
            opacity=0.6
        )
        
        fig.update_layout(
            template=TEMPLATE,
            paper_bgcolor=CORES['card_bg'],
            plot_bgcolor=CORES['card_bg'],
            font_color=CORES['text'],
            title_font_color=CORES['text'],
            title_x=0.5,
            height=500,
            xaxis=dict(gridcolor=CORES['border'], title_font_color=CORES['text_secondary']),
            yaxis=dict(gridcolor=CORES['border'], title_font_color=CORES['text_secondary'])
        )
        
        grafico = dcc.Graph(figure=fig, config={'displayModeBar': True, 'displaylogo': False})
    else:
        grafico = html.P("Dados insuficientes para gerar gráfico", style={'color': CORES['warning'], 'textAlign': 'center'})

    return html.Div([stats_cards, grafico])