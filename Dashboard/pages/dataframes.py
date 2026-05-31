# pages/dataframes.py
import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from data_loader import data_manager


# ================== CORES PADRÃO ==================
CORES = data_manager.get_cores()


# ================== PREPROCESSAMENTO ==================
def preprocess_data(df):
    """Pré-processamento usando o DataFrame já limpo do DataManager"""
    
    if data_manager.get_clean_df() is not None:
        return data_manager.get_clean_df().copy()
    
    df = df.copy()

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    if 'hrv' in df.columns:
        df['hrv'] = pd.to_numeric(df['hrv'], errors='coerce')
        df.loc[df['hrv'] > 500, 'hrv'] = pd.NA
        df.loc[df['hrv'] < 10, 'hrv'] = pd.NA

    numeric_cols = [
        'sleep_hours', 'recovery_score', 'day_strain', 
        'calories_burned', 'steps', 'resting_heart_rate'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'sleep_hours' in df.columns:
        df.loc[(df['sleep_hours'] < 2) | (df['sleep_hours'] > 16), 'sleep_hours'] = pd.NA

    for col in ['light_sleep_hours', 'deep_sleep_hours', 'rem_sleep_hours']:
        if col in df.columns:
            serie = pd.to_numeric(df[col], errors='coerce')
            if not serie.empty and serie.max() > 24:
                serie = serie / 60
            df[col] = serie
            df.loc[(df[col] < 0) | (df[col] > 12), col] = pd.NA

    return df


# ================== AUXILIAR ==================
def safe_graph(fig):
    return dcc.Graph(figure=fig) if fig else None


# ================== LAYOUT ==================
def create_layout(df):
    if df is None or df.empty:
        return html.Div("Sem dados", style={"color": "white"})

    df_clean = data_manager.get_clean_df()
    if df_clean is not None:
        df = df_clean.copy()
    else:
        df = preprocess_data(df)

    # ================== KPIs ==================
    total_users = df['user_id'].nunique() if 'user_id' in df.columns else 0
    total_records = len(df)
    
    total_workouts = 0
    if 'workout_completed' in df.columns:
        total_workouts = pd.to_numeric(df['workout_completed'], errors='coerce').fillna(0).sum()

    avg_recovery = df['recovery_score'].mean() if 'recovery_score' in df.columns else 0

    def card(title, value, color):
        return dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H4(f"{value}", className="card-title"),
                    html.P(title)
                ]),
                color=color,
                inverse=True
            ),
            md=3
        )

    cards = dbc.Row([
        card("Usuários", total_users, "secondary"),
        card("Registros", f"{total_records:,}", "info"),
        card("Treinos", f"{int(total_workouts):,}", "success"),
        card("Recovery Médio", f"{avg_recovery:.1f}", "warning"),
    ], className="mb-4")

    workout_df = df[df['workout_completed'] == 1] if 'workout_completed' in df.columns else df.copy()

    # ================== EVOLUÇÃO TEMPORAL ==================
    fig_time = None
    if {'date', 'recovery_score', 'day_strain', 'sleep_hours'}.issubset(df.columns):
        daily_avg = df.groupby('date')[['recovery_score', 'day_strain', 'sleep_hours']].mean().reset_index()
        
        fig_time = go.Figure()
        
        # Recovery Score - linha 1 (verde)
        fig_time.add_trace(go.Scatter(
            x=daily_avg['date'],
            y=daily_avg['recovery_score'],
            mode='lines',
            name=data_manager.traduzir_coluna('recovery_score'),
            line=dict(color=CORES['recovery'], width=2)
        ))
        
        # Day Strain - linha 2 (laranja)
        fig_time.add_trace(go.Scatter(
            x=daily_avg['date'],
            y=daily_avg['day_strain'],
            mode='lines',
            name=data_manager.traduzir_coluna('day_strain'),
            line=dict(color=CORES['strain'], width=2)
        ))
        
        # Sleep Hours - linha 3 (azul)
        fig_time.add_trace(go.Scatter(
            x=daily_avg['date'],
            y=daily_avg['sleep_hours'],
            mode='lines',
            name=data_manager.traduzir_coluna('sleep_hours'),
            line=dict(color=CORES['sleep'], width=2)
        ))
        
        fig_time.update_layout(
            template='plotly_dark',
            title='Evolução Temporal das Métricas',
            xaxis_title='Data',
            yaxis_title='Valor',
            paper_bgcolor=CORES['card_bg'],
            plot_bgcolor=CORES['card_bg'],
            font_color=CORES['text'],
            legend=dict(
                title='Métrica',
                bgcolor='rgba(0,0,0,0.6)',
                bordercolor=CORES['border']
            )
        )

    # ================== PARETO ATIVIDADES ==================
    fig_pareto = None
    if 'activity_type' in workout_df.columns:
        atividades_traduzidas = workout_df['activity_type'].apply(
            lambda x: data_manager.VALUE_TRANSLATIONS.get(str(x).lower(), x) if pd.notna(x) else x
        )
        
        pareto = atividades_traduzidas.value_counts().reset_index()
        pareto.columns = ['atividade', 'frequencia']
        pareto['percentual'] = pareto['frequencia'] / pareto['frequencia'].sum() * 100
        pareto['acumulado'] = pareto['percentual'].cumsum()

        fig_pareto = go.Figure()
        fig_pareto.add_bar(
            y=pareto['atividade'],
            x=pareto['frequencia'],
            name='Frequência',
            orientation='h',
            marker_color=CORES['intermediario']
        )
        fig_pareto.add_scatter(
            y=pareto['atividade'],
            x=pareto['acumulado'],
            name='% Acumulado',
            xaxis='x2',
            mode='lines+markers',
            line=dict(color=CORES['feminino'], width=3),
            marker=dict(color=CORES['feminino'], size=8)
        )
        fig_pareto.add_vline(x=80, line_dash='dash', line_color=CORES['danger'])
        fig_pareto.update_layout(
            template='plotly_dark',
            title='Pareto de Atividades (80/20)',
            xaxis=dict(title='Frequência'),
            xaxis2=dict(title='% Acumulado', overlaying='x', side='top', range=[0, 100]),
            paper_bgcolor=CORES['card_bg'],
            plot_bgcolor=CORES['card_bg'],
            font_color=CORES['text'],
            legend=dict(bgcolor='rgba(0,0,0,0.6)', bordercolor=CORES['border'])
        )
        fig_pareto.update_traces(marker_line_width=0)

    # ================== IMPACTO DOS TREINOS ==================
    fig_wr = None
    if {'activity_type', 'recovery_score', 'day_strain'}.issubset(workout_df.columns):
        wr = (workout_df.groupby('activity_type')
              .agg({'recovery_score': 'mean', 'day_strain': 'mean'})
              .sort_values('day_strain', ascending=False)
              .head(10)
              .reset_index())
        
        wr['activity_type'] = wr['activity_type'].apply(
            lambda x: data_manager.VALUE_TRANSLATIONS.get(str(x).lower(), x) if pd.notna(x) else x
        )

        fig_wr = go.Figure()
        fig_wr.add_bar(
            y=wr['activity_type'],
            x=wr['recovery_score'],
            name=data_manager.traduzir_coluna('recovery_score'),
            orientation='h',
            marker_color=CORES['sleep']
        )
        fig_wr.add_scatter(
            y=wr['activity_type'],
            x=wr['day_strain'],
            name=data_manager.traduzir_coluna('day_strain'),
            xaxis='x2',
            line=dict(color=CORES['strain'], width=3),
            marker=dict(color=CORES['strain'], size=8)
        )
        fig_wr.update_layout(
            template='plotly_dark',
            xaxis=dict(title=data_manager.traduzir_coluna('recovery_score')),
            xaxis2=dict(title=data_manager.traduzir_coluna('day_strain'), overlaying='x', side='top'),
            title='Impacto dos Treinos por Atividade',
            paper_bgcolor=CORES['card_bg'],
            plot_bgcolor=CORES['card_bg'],
            font_color=CORES['text'],
            legend=dict(bgcolor='rgba(0,0,0,0.6)', bordercolor=CORES['border'])
        )

    # ================== TREINOS POR DIA ==================
    fig_weekday = None
    if 'date' in df.columns:
        temp_df = df.assign(
            weekday=df['date'].dt.day_name().apply(
                lambda x: data_manager.VALUE_TRANSLATIONS.get(str(x).lower(), x) if pd.notna(x) else x
            )
        )
        
        if 'workout_completed' in temp_df.columns:
            weekday_counts = temp_df[temp_df['workout_completed'] == 1].groupby('weekday').size()
        else:
            weekday_counts = temp_df.groupby('weekday').size()
        
        dias_ordem = {'Segunda': 0, 'Terça': 1, 'Quarta': 2, 'Quinta': 3, 'Sexta': 4, 'Sábado': 5, 'Domingo': 6}
        weekday_counts = weekday_counts.reindex(sorted(weekday_counts.index, key=lambda x: dias_ordem.get(x, 7)))
        
        fig_weekday = px.bar(
            y=weekday_counts.index,
            x=weekday_counts.values,
            template='plotly_dark',
            title='Treinos por Dia da Semana',
            orientation='h',
            labels={'y': 'Dia da Semana', 'x': 'Quantidade de Treinos'},
            color_discrete_sequence=[CORES['chart_colors'][3]],
            text=weekday_counts.values
        )
        
        fig_weekday.update_traces(texttemplate='%{text}', textposition='outside')
        fig_weekday.update_layout(
            paper_bgcolor=CORES['card_bg'],
            plot_bgcolor=CORES['card_bg'],
            font_color=CORES['text']
        )

    # ================== PARETO USUÁRIOS ==================
    fig_users = None
    if {'user_id', 'workout_completed'}.issubset(df.columns):
        users = (df[df['workout_completed'] == 1]
                 .groupby('user_id')['workout_completed']
                 .sum()
                 .sort_values(ascending=False)
                 .reset_index())
        users.columns = ['usuario', 'treinos']
        users['percentual'] = users['treinos'] / users['treinos'].sum() * 100
        users['acumulado'] = users['percentual'].cumsum()

        fig_users = go.Figure()
        fig_users.add_bar(
            y=users['usuario'].astype(str),
            x=users['treinos'],
            name='Treinos',
            orientation='h',
            marker_color=CORES['feminino']
        )
        fig_users.add_scatter(
            y=users['usuario'].astype(str),
            x=users['acumulado'],
            name='% Acumulado',
            xaxis='x2',
            mode='lines+markers',
            line=dict(color=CORES['intermediario'], width=3),
            marker=dict(color=CORES['intermediario'], size=8)
        )
        fig_users.add_vline(x=80, line_dash='dash', line_color=CORES['danger'])
        fig_users.update_layout(
            template='plotly_dark',
            title='Pareto de Usuários (80/20)',
            xaxis=dict(title='Treinos'),
            xaxis2=dict(title='% Acumulado', overlaying='x', side='top', range=[0, 100]),
            paper_bgcolor=CORES['card_bg'],
            plot_bgcolor=CORES['card_bg'],
            font_color=CORES['text'],
            legend=dict(bgcolor='rgba(0,0,0,0.6)', bordercolor=CORES['border'])
        )

    # ================== MATRIZ DE CORRELAÇÃO ==================
    fig_corr = None
    ignore_cols = ['user_id', 'id', 'workout_id']
    numeric = df.select_dtypes(include=['float64', 'int64']).copy()
    numeric = numeric.drop(columns=[c for c in ignore_cols if c in numeric.columns], errors='ignore')
    numeric = numeric.loc[:, numeric.nunique() > 1]
    numeric = numeric.dropna(axis=1, thresh=len(numeric) * 0.5)

    if numeric.shape[1] >= 2:
        corr = numeric.corr()
        threshold = 0.5
        corr[corr.abs() < threshold] = 0
        
        top_cols = corr.abs().mean().sort_values(ascending=False).head(15).index
        corr = corr.loc[top_cols, top_cols]
        
        # Traduzir nomes das colunas
        colunas_traduzidas = [data_manager.traduzir_coluna(c) for c in corr.columns]
        corr.columns = colunas_traduzidas
        corr.index = colunas_traduzidas
        
        colorscale = [
            [0.0, CORES['danger']],
            [0.25, '#f46d43'],
            [0.5, '#fee090'],
            [0.75, '#abd9e9'],
            [1.0, CORES['success']]
        ]
        
        fig_corr = px.imshow(
            corr,
            text_auto='.2f',
            color_continuous_scale=colorscale,
            zmin=-1, zmax=1,
            aspect='auto'
        )
        fig_corr.update_layout(
            template='plotly_dark',
            title='Matriz de Correlação entre Variáveis',
            height=700,
            margin=dict(l=40, r=40, t=60, b=40),
            paper_bgcolor=CORES['card_bg'],
            plot_bgcolor=CORES['card_bg'],
            font_color=CORES['text'],
            coloraxis_colorbar=dict(
                title='Correlação',
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=['-1 (Negativa Forte)', '-0.5', '0', '0.5', '1 (Positiva Forte)']
            )
        )
        fig_corr.update_xaxes(tickangle=45)

    # ================== LAYOUT FINAL ==================
    return html.Div([
        dbc.Button(
            "Voltar", 
            href="/", 
            color="dark", 
            className="mb-3",
            style={'backgroundColor': 'transparent', 'border': f'1px solid {CORES["border"]}', 'color': CORES['text']}
        ),
        html.H1("Análise de Dados", className="text-light mb-4", style={'color': CORES['text']}),
        cards,
        safe_graph(fig_time),
        dbc.Row([
            dbc.Col(safe_graph(fig_pareto), md=6) if fig_pareto else None,
            dbc.Col(safe_graph(fig_wr), md=6) if fig_wr else None,
        ]),
        dbc.Row([
            dbc.Col(safe_graph(fig_weekday), md=6) if fig_weekday else None,
            dbc.Col(safe_graph(fig_users), md=6) if fig_users else None,
        ]),
        html.Hr(style={'borderColor': CORES['border']}),
        html.H4("Matriz de Correlação", className="text-light", style={'color': CORES['text']}),
        safe_graph(fig_corr) if fig_corr else html.P("Sem dados numéricos suficientes para correlação", className="text-light", style={'color': CORES['text_secondary']}),
    ], style={"backgroundColor": CORES['background'], "padding": "20px"})