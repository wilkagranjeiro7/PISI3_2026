import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# ==============================================================================
# 1. Configurações de Estilo e Paleta de Cores
# ==============================================================================
minha_paleta_azul = ['#1e3a8a', '#3b82f6', '#93c5fd']

estilo_dark = dict(
    paper_bgcolor='rgba(0,0,0,0)', 
    plot_bgcolor='rgba(0,0,0,0)',  
    font_color='#FFFFFF',          
    title_font_color='#60A5FA'     
)

# ==============================================================================
# 2. Abrir a maleta de dados
# ==============================================================================
caminho_pkl = 'lts.pkl'
if os.path.exists(caminho_pkl):
    dados = joblib.load(caminho_pkl)
    metrics = dados['metrics']
    dist_antes = dados['dist_antes']
    dist_depois = dados['dist_depois']
    y_test = dados.get('y_test', [])
    y_proba = dados.get('y_proba', [])
    classes = dados.get('classes', [])
else:
    metrics = {'accuracy': 0, 'f1_score': 0, 'conf_matrix': [[0]], 'classes': []}
    dist_antes = {}
    dist_depois = {}
    y_test, y_proba, classes = [], [], []

# ==============================================================================
# 3. Processamento dos Gráficos
# ==============================================================================

# Gráficos de Balanceamento
df_antes = pd.DataFrame(list(dist_antes.items()), columns=['Categoria', 'Quantidade'])
fig_antes = px.bar(df_antes, x='Categoria', y='Quantidade', title='Antes do Balanceamento', 
                    color='Categoria', text_auto='.0f', color_discrete_sequence=minha_paleta_azul)
fig_antes.update_layout(**estilo_dark).update_yaxes(gridcolor='#2A2A2A')

df_depois = pd.DataFrame(list(dist_depois.items()), columns=['Categoria', 'Quantidade'])
fig_depois = px.bar(df_depois, x='Categoria', y='Quantidade', title='Depois do SMOTE', 
                     color='Categoria', text_auto='.0f', color_discrete_sequence=minha_paleta_azul)
fig_depois.update_layout(**estilo_dark).update_yaxes(gridcolor='#2A2A2A')

# Comparação de Performance
acuracia_3va_lightgbm = 65.8 
df_comp = pd.DataFrame({
    'Modelo': ['Benchmark 3VA (LightGBM)', 'Seu Modelo (Random Forest)'],
    'Acurácia': [acuracia_3va_lightgbm, metrics['accuracy']*100] 
})
fig_comparativa = px.bar(
    df_comp, x='Modelo', y='Acurácia', title='Comparação de Performance', 
    color='Modelo', text_auto='.1f', color_discrete_sequence=['#475569', '#3b82f6']
)
fig_comparativa.update_layout(**estilo_dark)

# Curva ROC
if len(y_test) > 0 and len(classes) > 0:
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = y_test_bin.shape[1]
    fig_roc = go.Figure()
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{classes[i]} (AUC={roc_auc:.2f})'))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color='white', dash='dash'), name='Aleatório'))
    
    # Anotação técnica mínima mantida apenas na curva ROC
    fig_roc.add_annotation(x=0.75, y=0.15, text="Modelo > Aleatório", showarrow=False, font=dict(color="gray", size=10))
    
    fig_roc.update_layout(
        title='Curva ROC (Multiclasse)', **estilo_dark,
        margin=dict(l=50, r=50, t=60, b=50),
        legend=dict(orientation="v", yanchor="bottom", y=0.05, xanchor="right", x=0.95, bgcolor="rgba(0,0,0,0.5)")
    )
else:
    fig_roc = go.Figure().update_layout(title='Dados ROC indisponíveis', **estilo_dark)

# Matriz de Confusão
if len(metrics['classes']) > 0:
    fig_matriz = px.imshow(metrics['conf_matrix'], labels=dict(x="Previsão", y="Realidade", color="Atletas"), 
                           x=metrics['classes'], y=metrics['classes'], text_auto=True, color_continuous_scale='Blues', title='Matriz de Confusão')
else:
    fig_matriz = go.Figure()
fig_matriz.update_layout(**estilo_dark)

# ==============================================================================
# 4. Estrutura Visual do Dashboard (Layout Organizado)
# ==============================================================================
def create_layout(df):
    return dbc.Container([
        html.H2("Classificação Avançada: Previsão de Recovery", className="mt-4 mb-4 text-primary"),

        html.H4("1. Performance do Modelo", className="text-info mt-4"),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardHeader("Comparativo de Acurácia"), dbc.CardBody(dcc.Graph(figure=fig_comparativa))]), md=6),
            dbc.Col(dbc.Card([dbc.CardHeader("Curva ROC"), dbc.CardBody(dcc.Graph(figure=fig_roc))]), md=6),
        ], className="mb-4"),

        html.H4("2. Tratamento de Dados", className="text-info mt-4"),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardHeader("Distribuição Original"), dbc.CardBody(dcc.Graph(figure=fig_antes))]), md=6),
            dbc.Col(dbc.Card([dbc.CardHeader("Distribuição Pós-SMOTE"), dbc.CardBody(dcc.Graph(figure=fig_depois))]), md=6),
        ], className="mb-4"),

        html.H4("3. Matriz de Confusão", className="text-info mt-4"),
        dbc.Row([dbc.Col(dbc.Card([dbc.CardHeader("Acertos e Erros"), dbc.CardBody(dcc.Graph(figure=fig_matriz))]), md=12)], className="mb-4"),

        html.H4("4. Explicabilidade (XAI - SHAP)", className="text-info mt-4"),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Importância das Variáveis (Impacto Global)"),
                dbc.CardBody(html.Img(
                    src="/assets/shap_bar_plot.png", 
                    style={'width': '100%', 'borderRadius': '8px', 'padding': '10px'}
                ))
            ]), md=12)
        ], className="mb-4"),

    ], fluid=True)