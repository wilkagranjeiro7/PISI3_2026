# pages/boolean.py
from dash import html, dcc, Input, Output, callback, callback_context 
import plotly.express as px
import plotly.graph_objects as go

def create_layout(df):
    """Página de análise booleana com queries lógicas"""
    
    numeric_cols = [c for c in df.columns if df[c].dtype in ['int64', 'float64']]
    
    return html.Div([
        html.H1("Análise Boolean - Queries Lógicas", style={'marginBottom': 20}),
        
        html.Div([
            html.Div([
                html.Div("📋 Exemplos Rápidos", className="filter-label"),
                html.Div([
                    html.Button("Recovery Alto (>66)", id="query-ex1", 
                               style={'margin': '5px', 'padding': '10px', 'backgroundColor': '#2ecc71', 
                                      'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                    html.Button("Sono Ruim (<6h)", id="query-ex2", 
                               style={'margin': '5px', 'padding': '10px', 'backgroundColor': '#e74c3c', 
                                      'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                    html.Button("Treino Intenso (Strain>15)", id="query-ex3", 
                               style={'margin': '5px', 'padding': '10px', 'backgroundColor': '#f39c12', 
                                      'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                    html.Button("HRV Baixo (<50)", id="query-ex4", 
                               style={'margin': '5px', 'padding': '10px', 'backgroundColor': '#9b59b6', 
                                      'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                ])
            ], className="filter-group"),
            
            html.Div([
                html.Div("🔹 Condição 1:", className="filter-label", style={'fontWeight': 'bold'}),
                html.Div([
                    dcc.Dropdown(id='bool-col1', options=[{'label': c, 'value': c} for c in numeric_cols],
                                value='recovery_score', style={'width': '45%', 'display': 'inline-block', 'marginRight': '2%'}),
                    dcc.Dropdown(id='bool-op1', options=[{'label': op, 'value': op} for op in ['>', '<', '>=', '<=', '==', '!=']],
                                value='>', style={'width': '20%', 'display': 'inline-block', 'marginRight': '2%'}),
                    dcc.Input(id='bool-val1', type='number', value=66,
                             style={'width': '25%', 'display': 'inline-block', 'padding': '8px'})
                ])
            ], className="filter-group"),
            
            html.Div([
                html.Div("🔸 Operador Lógico:", className="filter-label", style={'fontWeight': 'bold'}),
                dcc.RadioItems(id='bool-logic', options=[{'label': ' AND (E)', 'value': 'and'}, {'label': ' OR (OU)', 'value': 'or'}],
                              value='and', labelStyle={'display': 'inline-block', 'marginRight': 20})
            ], className="filter-group"),
            
            html.Div([
                html.Div("🔹 Condição 2:", className="filter-label", style={'fontWeight': 'bold'}),
                html.Div([
                    dcc.Dropdown(id='bool-col2', options=[{'label': c, 'value': c} for c in numeric_cols],
                                value='sleep_hours', style={'width': '45%', 'display': 'inline-block', 'marginRight': '2%'}),
                    dcc.Dropdown(id='bool-op2', options=[{'label': op, 'value': op} for op in ['>', '<', '>=', '<=', '==', '!=']],
                                value='<', style={'width': '20%', 'display': 'inline-block', 'marginRight': '2%'}),
                    dcc.Input(id='bool-val2', type='number', value=7,
                             style={'width': '25%', 'display': 'inline-block', 'padding': '8px'})
                ])
            ], className="filter-group"),
            
            html.Div(id='bool-resultados', className="chart-card"),
            html.Div(id='bool-grafico', className="chart-card")
        ])
    ])

@callback(
    [Output('bool-resultados', 'children'),
     Output('bool-grafico', 'children')],
    [Input('bool-col1', 'value'), Input('bool-op1', 'value'), Input('bool-val1', 'value'),
     Input('bool-logic', 'value'),
     Input('bool-col2', 'value'), Input('bool-op2', 'value'), Input('bool-val2', 'value'),
     Input('query-ex1', 'n_clicks'), Input('query-ex2', 'n_clicks'),
     Input('query-ex3', 'n_clicks'), Input('query-ex4', 'n_clicks')]
)
def update_boolean(col1, op1, val1, logic, col2, op2, val2, ex1, ex2, ex3, ex4):
    from data_loader import data_manager
    df = data_manager.df
    
    ctx = callback_context  # Now this will work
    if not ctx.triggered or ctx.triggered[0]['prop_id'] == '.':
        query = f"{col1} {op1} {val1} {logic} {col2} {op2} {val2}"
    else:
        button = ctx.triggered[0]['prop_id'].split('.')[0]
        queries = {'query-ex1': 'recovery_score > 66', 'query-ex2': 'sleep_hours < 6',
                   'query-ex3': 'day_strain > 15', 'query-ex4': 'hrv < 50'}
        query = queries.get(button, f"{col1} {op1} {val1} {logic} {col2} {op2} {val2}")
    
    try:
        df_filtered = df.query(query)
        
        resultados = html.Div([
            html.H4(f"🔍 Query Executada: {query}", style={'marginBottom': 15}),
            html.Div([
                html.Div([html.Div(f"{len(df_filtered):,}", className="stat-number"), html.Div("Registros")], className="stat-card"),
                html.Div([html.Div(f"{len(df_filtered)/len(df)*100:.1f}%", className="stat-number"), html.Div("Percentual")], className="stat-card"),
                html.Div([html.Div(f"{df_filtered['recovery_score'].mean():.1f}", className="stat-number"), html.Div("Recovery Médio")], className="stat-card"),
            ], className="stats-grid")
        ])
        
        fig = go.Figure()
        fig.add_trace(go.Box(y=df_filtered['recovery_score'], name='Filtrado', marker_color='#2ecc71'))
        fig.add_trace(go.Box(y=df['recovery_score'], name='Total', marker_color='#e74c3c'))
        fig.update_layout(title='Comparação: Dados Filtrados vs Total', template='plotly_white', height=450)
        
        return resultados, dcc.Graph(figure=fig)
    except Exception as e:
        return html.Div(f"Erro na query: {str(e)}. Verifique os parâmetros."), html.Div()