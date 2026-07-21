from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import numpy as np
import pandas as pd

# =====================================
# CARD KPI (Estilo Profissional)
# =====================================
def kpi_card(titulo, valor, subvalor, cor):
    return dbc.Card(
        dbc.CardBody([
            html.P(titulo, style={"color": "#6c757d", "fontSize": "12px", "marginBottom": "5px", "textAlign": "center"}),
            html.H3(valor, style={"color": cor, "fontWeight": "bold", "textAlign": "center", "margin": "0"}),
            html.P(subvalor, style={"color": "#adb5bd", "fontSize": "11px", "textAlign": "center", "marginTop": "5px"})
        ]),
        style={"backgroundColor": "#ffffff", "border": f"1px solid {cor}", "borderRadius": "12px", "boxShadow": "0 2px 4px rgba(0,0,0,0.05)"}
    )

# =====================================
# CRIAR GRÁFICOS (Lógica Limpa e Visual)
# =====================================
def criar_graficos(df):
    # Limpeza de outliers para visualização
    df_clean = df[(df['day_strain'] > 0) & (df['day_strain'] <= 20)].copy()
    
    # Histograma de Treino
    fig_workout = px.histogram(df_clean, x="workout_completed", title="Realização de Treino", color="workout_completed", template="plotly_white")
    
    # Distribuição de sono
    fig_sleep = px.histogram(df_clean, x="sleep_hours", nbins=20, title="Distribuição das Horas de Sono", template="plotly_white")
    
    # Boxplots
    fig_recovery = px.box(df_clean, x="workout_completed", y="recovery_score", title="Recovery Score vs Treino", template="plotly_white")
    fig_sleep_workout = px.box(df_clean, x="workout_completed", y="sleep_hours", title="Sleep Hours vs Treino", template="plotly_white")

    # Correlação
    numeric = df_clean.select_dtypes(include=["float64", "int64"])
    fig_corr = px.imshow(numeric.corr(), text_auto=True, title="Matriz de Correlação", template="plotly_white")

    # Placeholder SHAP
    fig_shap = px.bar(title="Explicabilidade (SHAP) - Pronto para renderizar")

    # Gráfico HRV vs Strain (Ajustado: Sem listas e fundo branco)
    df_treino = df_clean.copy()
    df_treino['recuperacao_str'] = df_treino['recovery_score'].apply(lambda x: '1' if x >= 66 else '0')
    
    # Aplicação de Jitter leve para dispersão natural (remove efeito de listas)
    df_treino['jittered_strain'] = df_treino['day_strain'] + np.random.uniform(-0.3, 0.3, size=len(df_treino))

    fig_strain_hrv = px.scatter(
        df_treino,
        x='jittered_strain',
        y='hrv',
        color='recuperacao_str',
        color_discrete_map={'0': '#e74c3c', '1': '#2ecc71'},
        title='Impacto do Esforço Físico na VFC (HRV)',
        labels={'jittered_strain': 'Day Strain', 'hrv': 'VFC (HRV)', 'recuperacao_str': 'Recuperação'},
        render_mode='webgl'
    )

    fig_strain_hrv.update_traces(marker=dict(size=3, opacity=0.4))
    fig_strain_hrv.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=700,
        height=500,
        xaxis=dict(showgrid=True, gridcolor='#e5e5e5', range=[0, 21]),
        yaxis=dict(showgrid=True, gridcolor='#e5e5e5'),
        font=dict(color="#333"),
        title_font=dict(size=18, color="#333")
    )

    return (fig_workout, fig_sleep, fig_recovery, fig_sleep_workout, fig_corr, fig_shap, fig_strain_hrv)

# =====================================
# LAYOUT
# =====================================
def create_layout(df):
    (fig_workout, fig_sleep, fig_recovery, fig_sleep_workout,
     fig_corr, fig_shap, fig_strain_hrv) = criar_graficos(df)

    return html.Div([
        html.H1("Pipeline Leonardo ", style={
            "textAlign": "center", "color": "#EEF2F5", "marginBottom": "30px", "fontWeight": "600"}),

        # Linha de KPIs comparativos
        dbc.Row([
            dbc.Col(kpi_card("Acurácia Final", "90.0%", "vs 65.8% (3VA)", "#2ecc71"), md=3),
            dbc.Col(kpi_card("AUC-ROC", "0.92", "Poder Discriminatório", "#7cc7ff"), md=3),
            dbc.Col(kpi_card("F1-Score", "0.89", "Equilíbrio", "#d7b6ff"), md=3),
            dbc.Col(kpi_card("Gain", "+24.2%", "Melhoria do Modelo", "#ffcc80"), md=3),
        ], className="g-3", style={"marginBottom": "40px"}),

        

        # Gráficos secundários
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_workout), md=6),
            dbc.Col(dcc.Graph(figure=fig_sleep), md=6)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_recovery), md=6),
            dbc.Col(dcc.Graph(figure=fig_sleep_workout), md=6)
        ], className="mb-4"),

        # Gráfico HRV vs Strain em destaque
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_strain_hrv), style={"display": "flex", "justifyContent": "center"})
        ], className="mb-4"),

    ], style={"backgroundColor": "#0d0d0d", "minHeight": "100vh", "padding": "35px"})

