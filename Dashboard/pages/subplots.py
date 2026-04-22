# pages/subplots.py
from dash import html, dcc, Input, Output, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_layout(df):
    """Página de subplots - múltiplos gráficos"""
    
    return html.Div([
        html.H1("📊 Subplots - Múltiplos Gráficos", style={'marginBottom': 20}),
        
        html.Div([
            html.Div([
                html.H4("Análise Completa das Métricas", className="chart-title"),
                html.Div(id='subplots-grafico')
            ], className="chart-card")
        ])
    ])

@callback(
    Output('subplots-grafico', 'children'),
    Input('subplots-grafico', 'id')
)
def update_subplots(_):
    from data_loader import data_manager
    df = data_manager.df
    
    # Criar subplots 2x2
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Recovery Score', 'Day Strain', 'Sleep Hours', 'HRV'),
        specs=[[{'type': 'histogram'}, {'type': 'histogram'}],
               [{'type': 'box'}, {'type': 'box'}]]
    )
    
    # Histogramas
    fig.add_trace(go.Histogram(x=df['recovery_score'], name='Recovery', 
                               marker_color='#2ecc71', nbinsx=30), row=1, col=1)
    fig.add_trace(go.Histogram(x=df['day_strain'], name='Strain',
                               marker_color='#e74c3c', nbinsx=30), row=1, col=2)
    
    # Boxplots
    fig.add_trace(go.Box(y=df['sleep_hours'], name='Sono',
                         marker_color='#3498db'), row=2, col=1)
    fig.add_trace(go.Box(y=df['hrv'], name='HRV',
                         marker_color='#9b59b6'), row=2, col=2)
    
    fig.update_layout(title='Análise de Múltiplas Métricas',
                     template='plotly_white', height=700, showlegend=False)
    fig.update_xaxes(title_text="Valor", row=2, col=1)
    fig.update_xaxes(title_text="Valor", row=2, col=2)
    fig.update_yaxes(title_text="Frequência", row=1, col=1)
    fig.update_yaxes(title_text="Frequência", row=1, col=2)
    
    return dcc.Graph(figure=fig)