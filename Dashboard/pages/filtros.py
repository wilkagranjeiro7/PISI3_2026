# pages/filtros.py
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, callback, State
import plotly.express as px
import plotly.graph_objects as go

def create_layout(df):
    """Página com filtros interativos"""
    
    return html.Div([
        html.H1("🔍 Filtros Interativos", className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Filtros Disponíveis"),
                    dbc.CardBody([
                        html.Label("Nível de Fitness:"),
                        dcc.Dropdown(
                            id='filtro-fitness',
                            options=[{'label': 'Todos', 'value': 'Todos'}] + 
                                    [{'label': l, 'value': l} for l in df['fitness_level'].unique()],
                            value='Todos'
                        ),
                        html.Br(),
                        
                        html.Label("Gênero:"),
                        dcc.Dropdown(
                            id='filtro-genero',
                            options=[{'label': 'Todos', 'value': 'Todos'}] + 
                                    [{'label': g, 'value': g} for g in df['gender'].unique()],
                            value='Todos'
                        ),
                        html.Br(),
                        
                        html.Label("Esporte Principal:"),
                        dcc.Dropdown(
                            id='filtro-esporte',
                            options=[{'label': 'Todos', 'value': 'Todos'}] + 
                                    [{'label': s, 'value': s} for s in df['primary_sport'].unique()],
                            value='Todos'
                        ),
                        html.Br(),
                        
                        html.Label("Range de Recovery Score:"),
                        dcc.RangeSlider(
                            id='filtro-recovery',
                            min=0, max=100, step=1,
                            marks={0: '0', 25: '25', 50: '50', 75: '75', 100: '100'},
                            value=[0, 100]
                        ),
                        html.Br(),
                        
                        html.Label("Range de Horas de Sono:"),
                        dcc.RangeSlider(
                            id='filtro-sono',
                            min=0, max=12, step=0.5,
                            marks={0: '0', 3: '3', 6: '6', 9: '9', 12: '12'},
                            value=[0, 12]
                        ),
                    ])
                ])
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Resultados dos Filtros"),
                    dbc.CardBody([
                        html.Div(id='filtro-resultados'),
                        html.Hr(),
                        dcc.Graph(id='filtro-grafico')
                    ])
                ])
            ], width=9)
        ])
    ])

@callback(
    [Output('filtro-resultados', 'children'),
     Output('filtro-grafico', 'figure')],
    [Input('filtro-fitness', 'value'),
     Input('filtro-genero', 'value'),
     Input('filtro-esporte', 'value'),
     Input('filtro-recovery', 'value'),
     Input('filtro-sono', 'value')]
)
def aplicar_filtros(fitness, genero, esporte, recovery_range, sono_range):
    from data_loader import data_manager
    df = data_manager.df.copy()
    
    # Aplicar filtros
    if fitness != 'Todos':
        df = df[df['fitness_level'] == fitness]
    if genero != 'Todos':
        df = df[df['gender'] == genero]
    if esporte != 'Todos':
        df = df[df['primary_sport'] == esporte]
    
    df = df[(df['recovery_score'] >= recovery_range[0]) & 
            (df['recovery_score'] <= recovery_range[1])]
    df = df[(df['sleep_hours'] >= sono_range[0]) & 
            (df['sleep_hours'] <= sono_range[1])]
    
    # Resultados
    resultados = html.Div([
        html.H5(f"📊 Registros encontrados: {len(df):,}"),
        html.P(f"👥 Usuários: {df['user_id'].nunique()}"),
        html.P(f"📅 Período: {df['date'].min().date()} a {df['date'].max().date()}"),
        html.P(f"💚 Recovery médio: {df['recovery_score'].mean():.1f}"),
        html.P(f"😴 Sono médio: {df['sleep_hours'].mean():.1f}h")
    ])
    
    # Gráfico
    daily_avg = df.groupby('date').agg({
        'recovery_score': 'mean',
        'day_strain': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_avg['date'], y=daily_avg['recovery_score'],
                            mode='lines', name='Recovery Score'))
    fig.add_trace(go.Scatter(x=daily_avg['date'], y=daily_avg['day_strain'],
                            mode='lines', name='Day Strain'))
    fig.update_layout(title='Métricas Filtradas', template='plotly_white')
    
    return resultados, fig