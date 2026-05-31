import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def create_layout(df):
    """Página inicial com visão geral"""
    
    cards = dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4(f"{df['user_id'].nunique()}", className="card-title"),
                html.P("Usuários", className="card-text"),
            ])
        ], color="primary", inverse=True), width=3),
        
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4(f"{len(df):,}", className="card-title"),
                html.P("Registros", className="card-text"),
            ])
        ], color="success", inverse=True), width=3),
        
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4(f"{df['workout_completed'].sum():,}", className="card-title"),
                html.P("Treinos", className="card-text"),
            ])
        ], color="info", inverse=True), width=3),
        
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4(f"{df['recovery_score'].mean():.1f}", className="card-title"),
                html.P("Recovery Médio", className="card-text"),
            ])
        ], color="warning", inverse=True), width=3),
    ], className="mb-4")
    
    # Gráfico de evolução temporal
    daily_avg = df.groupby('date').agg({
        'recovery_score': 'mean',
        'day_strain': 'mean',
        'sleep_hours': 'mean'
    }).reset_index()
    
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(x=daily_avg['date'], y=daily_avg['recovery_score'],
                                  mode='lines', name='Recovery Score'))
    fig_time.add_trace(go.Scatter(x=daily_avg['date'], y=daily_avg['day_strain'],
                                  mode='lines', name='Day Strain'))
    fig_time.add_trace(go.Scatter(x=daily_avg['date'], y=daily_avg['sleep_hours'],
                                  mode='lines', name='Sleep Hours'))
    fig_time.update_layout(title='Evolução Temporal das Métricas',
                          xaxis_title='Data', yaxis_title='Valor',
                          template='plotly_white')
    
    # Gráfico de distribuição de atividades
    workout_df = df[df['workout_completed'] == 1]
    sport_counts = workout_df['activity_type'].value_counts().head(10)
    
    fig_sports = px.bar(x=sport_counts.values, y=sport_counts.index, orientation='h',
                        title='Top 10 Atividades',
                        labels={'x': 'Quantidade', 'y': 'Atividade'})
    
    # Opção 1: Gráfico de Treinos vs Recovery Score
    workout_recovery = df[df['workout_completed'] == 1].groupby('activity_type').agg({
        'recovery_score': 'mean',
        'day_strain': 'mean'
    }).reset_index().head(10)
    
    fig_workout_recovery = go.Figure()
    fig_workout_recovery.add_trace(go.Bar(
        x=workout_recovery['activity_type'],
        y=workout_recovery['recovery_score'],
        name='Recovery Score',
        marker_color='lightblue'
    ))
    fig_workout_recovery.add_trace(go.Scatter(
        x=workout_recovery['activity_type'],
        y=workout_recovery['day_strain'],
        name='Day Strain',
        yaxis='y2',
        marker=dict(symbol='circle', size=10, color='red')
    ))
    fig_workout_recovery.update_layout(
        title='Impacto dos Treinos na Recuperação',
        xaxis_title='Atividade',
        yaxis_title='Recovery Score',
        yaxis2=dict(title='Day Strain', overlaying='y', side='right'),
        template='plotly_white',
        hovermode='x unified'
    )
    
    # Opção 4: Heatmap de Atividade por Dia da Semana
    df_copy = df.copy()  # Evitar modificar o DataFrame original
    df_copy['date'] = pd.to_datetime(df_copy['date'])
    df_copy['weekday'] = df_copy['date'].dt.day_name()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_activity = df_copy[df_copy['workout_completed'] == 1].groupby('weekday').size().reindex(weekday_order).fillna(0)
    
    fig_weekday = px.bar(x=weekday_activity.index, y=weekday_activity.values,
                         title='Treinos por Dia da Semana',
                         labels={'x': 'Dia', 'y': 'Número de Treinos'},
                         color=weekday_activity.values,
                         color_continuous_scale='Viridis')
    fig_weekday.update_layout(template='plotly_white')
    
    # Opção 5: Gauge Chart - Recovery Score Médio
    avg_recovery = df['recovery_score'].mean()
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg_recovery,
        title={'text': "Recovery Score Médio"},
        delta={'reference': 70},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "red"},
                {'range': [33, 66], 'color': "yellow"},
                {'range': [66, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': avg_recovery
            }
        }
    ))
    fig_gauge.update_layout(height=400, template='plotly_white')
    
    # Layout da página - Agora com 3 colunas na segunda linha
    layout = html.Div([
        html.H1("🏠 Dashboard WHOOP Fitness", className="mb-4"),
        cards,
        
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_time), width=12, className="mb-4")
        ]),
        
        # Primeira linha de gráficos: Atividades e Treinos vs Recovery
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_sports), width=6),
            dbc.Col(dcc.Graph(figure=fig_workout_recovery), width=6)
        ], className="mb-4"),
        
        # Segunda linha de gráficos: Dias da semana e Gauge
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_weekday), width=6),
            dbc.Col(dcc.Graph(figure=fig_gauge), width=6)
        ], className="mb-4"),
        
        html.Hr(),
        
    ])
    
    return layout