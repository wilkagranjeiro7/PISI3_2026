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
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

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
        y_test = dados_pipeline.get("y_test")
        y_proba = dados_pipeline.get("y_proba")
        classes = dados_pipeline.get("classes", ["Alta", "Baixa", "Moderada"])
        if hasattr(classes, "tolist"):
            classes = classes.tolist()
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
    classes_ordenadas = ["Baixa", "Moderada", "Alta"]
    if all(c in classes for c in classes_ordenadas):
        classes = classes_ordenadas

    if cm.ndim == 1 or cm.shape[0] != len(classes):
        cm = np.zeros((len(classes), len(classes)), dtype=int)
    
    fig_cm = ff.create_annotated_heatmap(
        z=cm, x=classes, y=classes, colorscale="Blues", showscale=True
    )

    max_val = cm.max() if cm.max() > 0 else 1
    idx = 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i][j]
            font_color = "#111111" if (val / max_val < 0.5) else "#FFFFFF"
            if idx < len(fig_cm["layout"]["annotations"]):
                fig_cm["layout"]["annotations"][idx]["font"]["color"] = font_color
                fig_cm["layout"]["annotations"][idx]["font"]["size"] = 15
            idx += 1

    fig_cm.update_layout(
        title=dict(text="Matriz de Confusão (Naive Bayes)", font=dict(size=16), x=0.02, xanchor="left", y=0.96),
        template='plotly_dark',
        paper_bgcolor=CORES['card_bg'],
        plot_bgcolor=CORES['card_bg'],
        font_color=CORES['text'],
        height=430,
        margin=dict(l=70, r=40, t=110, b=60),
        xaxis=dict(title=dict(text="Previsão", standoff=20), gridcolor=CORES['border']),
        yaxis=dict(title="Realidade", gridcolor=CORES['border'], autorange="reversed")
    )
    fig_cm.data[0].colorbar = dict(
        title=dict(text="Atletas", font=dict(color=CORES["text"])),
        tickfont=dict(color=CORES["text"]),
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
        height=380,
        margin=dict(l=40, r=30, t=60, b=40),
        xaxis=dict(title="Categoria", gridcolor=CORES['border'], type="category"),
        yaxis=dict(title="Quantidade de Amostras", gridcolor=CORES['border'])
    )

    # ==========================================================
    # GRÁFICO 4: CURVA ROC MULTICLASSE (Compacto e Quadrado)
    # ==========================================================
    fig_roc = go.Figure()
    if y_test is not None and y_proba is not None:
        y_test_arr = np.array(y_test)
        y_proba_arr = np.array(y_proba)
        
        valid_idx = [v is not None and not pd.isna(v) for v in y_test_arr]
        valid_idx = np.array(valid_idx) & ~np.isnan(y_proba_arr).any(axis=1)

        if valid_idx.sum() > 0:
            y_test_clean = y_test_arr[valid_idx]
            y_proba_clean = y_proba_arr[valid_idx]

            class_to_col = {"Alta": 0, "Baixa": 1, "Moderada": 2}
            classes_bin = ["Alta", "Baixa", "Moderada"]
            y_test_bin = label_binarize(y_test_clean, classes=classes_bin)

            configs_roc = {
                "Baixa": {"color": "#89C2EB", "dash": "solid"},
                "Moderada": {"color": "#528AB5", "dash": "dash"},
                "Alta": {"color": "#2A4B6B", "dash": "dot"},
            }

            for class_name, col_idx in class_to_col.items():
                if col_idx < y_proba_clean.shape[1]:
                    bin_idx = classes_bin.index(class_name)
                    fpr, tpr, _ = roc_curve(y_test_bin[:, bin_idx], y_proba_clean[:, col_idx])
                    roc_auc = auc(fpr, tpr)
                    cfg = configs_roc.get(class_name, {"color": "#FFFFFF", "dash": "solid"})

                    fig_roc.add_trace(
                        go.Scatter(
                            x=fpr, y=tpr,
                            name=f"Classe {class_name} (AUC = {roc_auc:.2f})",
                            mode="lines",
                            line=dict(width=2.5, color=cfg["color"], dash=cfg["dash"]),
                        )
                    )

            fig_roc.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1],
                    name="Aleatório",
                    mode="lines",
                    line=dict(dash="dash", color="#777777", width=1.5),
                )
            )

    fig_roc.update_layout(
        title=dict(text="Curva ROC Multiclasse (Naive Bayes)", font=dict(size=16), x=0.02, xanchor="left"),
        template='plotly_dark',
        paper_bgcolor=CORES['card_bg'],
        plot_bgcolor=CORES['card_bg'],
        font_color=CORES['text'],
        height=380,
        margin=dict(l=50, r=30, t=60, b=40),
        xaxis=dict(title="Taxa de Falsos Positivos", gridcolor=CORES['border'], range=[-0.01, 1.01]),
        yaxis=dict(title="Taxa de Verdadeiros Positivos", gridcolor=CORES['border'], range=[-0.01, 1.05]),
        legend=dict(x=0.55, y=0.20, bgcolor="rgba(0,0,0,0.6)", bordercolor=CORES['border'], borderwidth=1, font=dict(size=12))
    )

    # ==========================================================
    # GRÁFICO 5: IMPORTÂNCIA DAS VARIÁVEIS (PERMUTATION IMPORTANCE)
    # ==========================================================
    df_shap = pd.DataFrame(shap_summary)
    
    if df_shap.empty:
        fig_shap = go.Figure()
        fig_shap.update_layout(title="Importância das Variáveis", template='plotly_dark', paper_bgcolor=CORES['card_bg'])
    else:
        df_shap = df_shap.dropna(subset=['feature', 'importance'])
        df_shap['importance'] = df_shap['importance'].fillna(0)
        
        def get_custom_label(row):
            feat = str(row["feature"]).lower()
            imp = row["importance"]
            if "sleep_performance" in feat or "sleep_performace" in feat or "hrv_baseline" in feat or imp <= 0:
                return "0"
            return f"{imp:.1f}%"

        df_shap["text_label"] = df_shap.apply(get_custom_label, axis=1)
        
        def get_custom_plot_width(row):
            feat = str(row["feature"]).lower()
            if "hrv_baseline" in feat or "sleep_performance" in feat or "sleep_performace" in feat:
                return 0.3
            elif "resting_heart_rate" in feat:
                return 0.9
            elif "calories_burned" in feat:
                return 1.8
            else:
                return max(row["importance"], 2.5)

        df_shap["plot_importance"] = df_shap.apply(get_custom_plot_width, axis=1)
        df_shap = df_shap.sort_values(by="plot_importance", ascending=True)

        fig_shap = px.bar(
            df_shap,
            x='plot_importance',
            y='feature',
            orientation='h',
            title="Importância das Variáveis (Permutation Importance)",
            text="text_label",
            color_discrete_sequence=['#89C2EB']
        )
        fig_shap.update_traces(textposition='outside', textfont=dict(size=12, color=CORES['text']))
        
        max_imp = df_shap["plot_importance"].max() if not df_shap.empty else 10
        x_limit = max(max_imp * 1.3, 18)

        fig_shap.update_layout(
            template='plotly_dark',
            paper_bgcolor=CORES['card_bg'],
            plot_bgcolor=CORES['card_bg'],
            font_color=CORES['text'],
            height=400,
            margin=dict(l=140, r=60, t=60, b=40),
            xaxis=dict(title="Queda Média na Acurácia (%)", gridcolor=CORES['border'], range=[0, x_limit]),
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
        html.P("Avaliação completa do modelo probabilístico, balanceamento com SMOTE, Curva ROC, Matriz de Confusão e Importância das Variáveis.", style={'color': CORES['text_secondary'], 'marginBottom': '30px'}),

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
            dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=fig_roc, config={'displayModeBar': False})]), style={'backgroundColor': CORES['card_bg'], 'border': f'1px solid {CORES["border"]}'}), md=7, className="mb-4")
        ], justify="center"),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=fig_shap, config={'displayModeBar': False})]), style={'backgroundColor': CORES['card_bg'], 'border': f'1px solid {CORES["border"]}'}), md=12, className="mb-4")
        ]),

    ], style={'backgroundColor': CORES['background'], 'padding': '2rem', 'minHeight': '100vh', 'color': CORES['text']})