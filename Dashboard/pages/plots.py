# pages/plots.py
from dash import html, dcc, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go

def create_layout(df):
    """Página de visualizações interativas"""
    
    numeric_cols = [c for c in df.columns if df[c].dtype in ['int64', 'float64']]
    
    return html.Div([
        html.H1("📉 Visualizações Interativas", style={'marginBottom': 20}),
        
        html.Div([
            html.Div([
                html.Div("⚙️ Configurações do Gráfico", className="filter-label", style={'fontSize': 16, 'fontWeight': 'bold'}),
                
                html.Div("Tipo de Gráfico:", className="filter-label", style={'marginTop': 10}),
                dcc.Dropdown(
                    id='plot-type',
                    options=[
                        {'label': '📈 Dispersão (Scatter)', 'value': 'scatter'},
                        {'label': '📊 Histograma', 'value': 'histogram'},
                        {'label': '📦 Boxplot', 'value': 'box'},
                        {'label': '🎻 Violin Plot', 'value': 'violin'},
                        {'label': '🔵 Barra (Bar)', 'value': 'bar'}
                    ],
                    value='scatter'
                ),
                
                html.Div("Eixo X:", className="filter-label", style={'marginTop': 10}),
                dcc.Dropdown(id='plot-x', options=[{'label': c, 'value': c} for c in numeric_cols], value='recovery_score'),
                
                html.Div("Eixo Y:", className="filter-label", style={'marginTop': 10}),
                dcc.Dropdown(id='plot-y', options=[{'label': c, 'value': c} for c in numeric_cols], value='day_strain'),
                
                html.Div("Cor (opcional):", className="filter-label", style={'marginTop': 10}),
                dcc.Dropdown(id='plot-color', options=[{'label': 'Nenhum', 'value': 'none'}] + 
                            [{'label': c, 'value': c} for c in ['fitness_level', 'gender', 'recovery_level', 'sleep_quality']],
                            value='none'),
            ], className="filter-group", style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                html.Div(id='plot-grafico', className="chart-card")
            ], style={'width': '73%', 'display': 'inline-block', 'marginLeft': '2%'})
        ])
    ])

@callback(
    [Output('plot-x', 'options'),
     Output('plot-y', 'options')],
    Input('plot-type', 'value')
)
def update_plot_options(_):
    from data_loader import data_manager
    df = data_manager.df
    numeric_cols = [{'label': c, 'value': c} for c in df.select_dtypes(include=['int64', 'float64']).columns]
    return numeric_cols, numeric_cols

@callback(
    Output('plot-grafico', 'children'),
    [Input('plot-type', 'value'),
     Input('plot-x', 'value'),
     Input('plot-y', 'value'),
     Input('plot-color', 'value')]
)
def update_plot(plot_type, x_col, y_col, color_col):
    from data_loader import data_manager
    df = data_manager.df
    
    color = None if color_col == 'none' else color_col
    
    if plot_type == 'scatter':
        fig = px.scatter(df, x=x_col, y=y_col, color=color,
                        title=f'{x_col} vs {y_col}',
                        color_continuous_scale='Viridis',
                        opacity=0.6)
    elif plot_type == 'histogram':
        fig = px.histogram(df, x=x_col, color=color, nbins=30,
                          title=f'Distribuição de {x_col}')
    elif plot_type == 'box':
        fig = px.box(df, y=x_col, color=color,
                    title=f'Boxplot de {x_col}')
    elif plot_type == 'violin':
        fig = px.violin(df, y=x_col, color=color, box=True,
                       title=f'Violin Plot de {x_col}')
    else:  # bar
        agg_df = df.groupby(x_col)[y_col].mean().reset_index()
        fig = px.bar(agg_df, x=x_col, y=y_col,
                    title=f'Média de {y_col} por {x_col}')
    
    fig.update_layout(template='plotly_white', height=550)
    return dcc.Graph(figure=fig)