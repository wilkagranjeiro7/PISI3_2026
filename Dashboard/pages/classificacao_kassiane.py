import os
import sys

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
sys.path.insert(0, ROOT_DIR)

import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib

from data_loader import data_manager

CORES = data_manager.get_cores()

def create_layout(df):
    # ==========================================================
    # CARREGAR DADOS DO PIPELINE (LTS.PKL)
    # ==========================================================
    try:
        dados_pipeline = joblib.load("lts.pkl")
        metrics_nb = dados_pipeline.get("metrics", {})
        metrics_lgbm = dados_pipeline.get("metrics_lgbm", {})
        dist_antes = dados_pipeline.get("dist_antes", {})
        dist_depois = dados_pipeline.get("dist_depois", {})
        shap_summary = dados_pipeline.get("shap_summary", [])
    except Exception as e:
        return html.Div([
            html.H3("⚠️ Arquivo lts.pkl não encontrado!", style={'color': CORES['danger']}),
            html.P(f"Execute o script 'train_pipeline.py' primeiro para gerar os dados. Erro: {e}", style={'color': CORES['text']})
        ], style={'padding': '3rem'})

    # ==========================================================
    # GRÁFICO 1: COMPARAÇÃO DE DESEMPENHO (NAIVE BAYES VS LIGHTGBM)
    # ==========================================================
    df_comparacao = pd.DataFrame({
        'Modelo': ['Naive Bayes (Atual)', 'LightGBM (3VA - Melhor)'],
        'Acurácia': [metrics_nb.get('accuracy', 0), metrics_lgbm.get('accuracy', 0)],
        'F1-Score': [metrics_nb.get('f1_score', 0), metrics_lgbm.get('f1_score', 0)],
        'Precision': [metrics_nb.get('precision', 0), metrics_lgbm.get('precision', 0)],
        'Recall': [metrics_nb.get('recall', 0), metrics_lgbm.get('recall', 0)]
    })

    fig_comp = px.bar(
        df_comparacao.melt(id_vars='Modelo', var_name='Métrica', value_name='Valor'),
        x='Métrica',
        y='Valor',
        color='Modelo',
        barmode='group',
        title="Comparação de Desempenho: Naive Bayes vs LightGBM (3VA)",
        color_discrete_map={'Naive Bayes (Atual)': '#89C2EB', 'LightGBM (3VA - Melhor)': '#2A4B6B'}
    )
    fig_comp.update_layout(
        template='plotly_dark',
        paper_bgcolor=CORES['card_bg'],
        plot_bgcolor=CORES['card_bg'],
        font_color=CORES['text'],
        height=320,
        yaxis=dict(range=[0, 1.05], gridcolor=CORES['border'])
    )

    # ==========================================================
    # GRÁFICO 2: MATRIZ DE CONFUSÃO (NAIVE BAYES)
    # ==========================================================
    cm = np.array(metrics_nb.get('conf_matrix', [[0]]), dtype=int)
    classes = metrics_nb.get('classes', ['Baixa', 'Moderada', 'Alta'])
    if hasattr(classes, "tolist"):
        classes = classes.tolist()

    if cm.ndim == 1 or cm.shape[0] != len(classes):
        cm = np.zeros((len(classes), len(classes)), dtype=int)
    
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=classes,
        y=classes,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 14, "color": CORES['text']},
        colorscale=[[0, '#1A1A1A'], [1, '#528AB5']],
        showscale=False
    ))
    fig_cm.update_layout(
        title="Matriz de Confusão (Naive Bayes)",
        template='plotly_dark',
        paper_bgcolor=CORES['card_bg'],
        plot_bgcolor=CORES['card_bg'],
        font_color=CORES['text'],
        height=300,
        xaxis=dict(title="Predito", gridcolor=CORES['border']),
        yaxis=dict(title="Real", gridcolor=CORES['border'], autorange="reversed")
    )

    # ==========================================================
    # GRÁFICO 3: BALANCEAMENTO COM SMOTE
    # ==========================================================
    df_smote = pd.DataFrame([
        {'Categoria': k, 'Quantidade': v, 'Fase': 'Antes do SMOTE'} for k, v in dist_antes.items()
    ] + [
        {'Categoria': k, 'Quantidade': v, 'Fase': 'Depois do SMOTE'} for k, v in dist_depois.items()
    ])

    fig_smote = px.bar(
        df_smote,
        x='Categoria',
        y='Quantidade',
        color='Fase',
        barmode='group',
        title="Impacto do Balanceamento (SMOTE) nas Classes de Recuperação",
        color_discrete_map={'Antes do SMOTE': '#528AB5', 'Depois do SMOTE': '#2A4B6B'}
    )
    fig_smote.update_layout(
        template='plotly_dark',
        paper_bgcolor=CORES['card_bg'],
        plot_bgcolor=CORES['card_bg'],
        font_color=CORES['text'],
        height=400,
        xaxis=dict(title="Categoria de Recuperação", gridcolor=CORES['border']),
        yaxis=dict(title="Quantidade de Amostras", gridcolor=CORES['border'])
    )

    # ==========================================================
    # ==========================================================
    # GRÁFICO 4: IMPORTÂNCIA DAS VARIÁVEIS (PERMUTATION IMPORTANCE)
    # ==========================================================
    df_shap = pd.DataFrame(shap_summary)
    
    if df_shap.empty:
        fig_shap = go.Figure()
        fig_shap.update_layout(title="Importância das Variáveis (Execute o pipeline primeiro)", template='plotly_dark', paper_bgcolor=CORES['card_bg'])
    else:
        df_shap = df_shap.dropna(subset=['feature', 'importance'])
        df_shap = df_shap.sort_values('importance', ascending=True)
        fig_shap = px.bar(
            df_shap,
            x='importance',
            y='feature',
            orientation='h',
            title="Importância das Variáveis (Permutation Importance)",
            color='importance',
            color_continuous_scale=['#1A1A1A', '#89C2EB']
        )
        fig_shap.update_layout(
            template='plotly_dark',
            paper_bgcolor=CORES['card_bg'],
            plot_bgcolor=CORES['card_bg'],
            font_color=CORES['text'],
            height=320,
            xaxis=dict(title="Queda Média na Acurácia (%)", gridcolor=CORES['border']),
            yaxis=dict(title="", gridcolor=CORES['border']),
            showlegend=False
        )

    # ==========================================================
    # LAYOUT DA PÁGINA
    # ==========================================================
    return html.Div([
        dbc.Button("← Voltar", href="/", color="light", size="sm",
                   style={'backgroundColor': 'transparent', 'border': f'1px solid {CORES["border"]}', 'color': CORES['text'], 'marginBottom': '20px'}),
        
        html.H2("Pipeline de Classificação: Naive Bayes & Explicabilidade", style={'color': CORES['text'], 'marginBottom': '10px'}),
        html.P("Avaliação completa do modelo probabilístico, balanceamento com SMOTE e comparação com a 3VA.", style={'color': CORES['text_secondary'], 'marginBottom': '30px'}),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H4(f"{metrics_nb.get('accuracy', 0):.1%}", style={'color': '#89C2EB'}),
                html.P("Acurácia (Naive Bayes)", style={'color': CORES['text_secondary'], 'fontSize': '12px'})
            ]), style={'backgroundColor': CORES['card_bg'], 'border': f'1px solid {CORES["border"]}'}), md=3),
            
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H4(f"{metrics_nb.get('f1_score', 0):.3f}", style={'color': '#528AB5'}),
                html.P("F1-Score Ponderado", style={'color': CORES['text_secondary'], 'fontSize': '12px'})
            ]), style={'backgroundColor': CORES['card_bg'], 'border': f'1px solid {CORES["border"]}'}), md=3),
            
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H4(f"{metrics_lgbm.get('accuracy', 0):.1%}", style={'color': '#2A4B6B'}),
                html.P("Acurácia (LightGBM - 3VA)", style={'color': CORES['text_secondary'], 'fontSize': '12px'})
            ]), style={'backgroundColor': CORES['card_bg'], 'border': f'1px solid {CORES["border"]}'}), md=3),
            
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H4("3 Classes", style={'color': '#FFFFFF'}),
                html.P("Target (Baixa, Moderada, Alta)", style={'color': CORES['text_secondary'], 'fontSize': '12px'})
            ]), style={'backgroundColor': CORES['card_bg'], 'border': f'1px solid {CORES["border"]}'}), md=3),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_comp, config={'displayModeBar': False}), md=12, className="mb-4")
        ]),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_smote, config={'displayModeBar': False}), md=6, className="mb-4"),
            dbc.Col(dcc.Graph(figure=fig_cm, config={'displayModeBar': False}), md=6, className="mb-4"),
        ]),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_shap, config={'displayModeBar': False}), md=12, className="mb-4")
        ]),

    ], style={'backgroundColor': CORES['background'], 'padding': '2rem', 'minHeight': '100vh', 'color': CORES['text']})