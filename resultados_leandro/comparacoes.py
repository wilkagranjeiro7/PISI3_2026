# ==================================================
# pages/comparacoes.py - COMPARAÇÃO DE MODELOS COM XAI (VERSÃO FINAL)
# ==================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, callback, State
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_loader import data_manager
from model_manager import model_manager
import warnings
warnings.filterwarnings('ignore')

# ==================================================
# TENTAR IMPORTAR SHAP
# ==================================================

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ==================================================
# CORES
# ==================================================

CORES = data_manager.get_cores()

# ==================================================
# FEATURES DISPONÍVEIS
# ==================================================

FEATURES_PADRAO = ['hrv', 'resting_heart_rate', 'day_strain', 'sleep_hours', 'sleep_efficiency']
FEATURES_DERIVADAS = ['sleep_quality', 'strain_per_sleep', 'hrv_ratio', 'hrv_rhr_ratio']
TODAS_FEATURES = FEATURES_PADRAO + FEATURES_DERIVADAS

# ==================================================
# FUNÇÕES SHAP
# ==================================================

def calcular_shap_para_modelo(model, X_sample, feature_names, modelo_tipo='classificacao'):
    """
    Calcula SHAP para um modelo (classificação ou regressão)
    """
    if not SHAP_AVAILABLE:
        return None, None
    
    try:
        if modelo_tipo == 'classificacao':
            if hasattr(model, 'predict_proba') and hasattr(model, 'feature_importances_'):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) >= 2 else shap_values[0]
                
                return shap_values, explainer
        else:
            if hasattr(model, 'feature_importances_'):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                return shap_values, explainer
        
        return None, None
    except Exception as e:
        print(f"Erro SHAP: {e}")
        return None, None


def criar_grafico_shap_importance(shap_values, feature_names, titulo, cores):
    """Cria gráfico de importância SHAP"""
    if shap_values is None:
        return None
    
    shap_importance = np.abs(shap_values).mean(axis=0)
    feature_names_traduzidos = [data_manager.traduzir_coluna(f) for f in feature_names]
    
    df_importance = pd.DataFrame({
        'Feature': feature_names_traduzidos,
        'Importance': shap_importance
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(
        df_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title=titulo,
        color='Importance',
        color_continuous_scale=['#A8B5C0', cores['accent']],
        text='Importance'
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=cores['card_bg'],
        plot_bgcolor=cores['card_bg'],
        font_color=cores['text'],
        height=350,
        xaxis=dict(title="Importância SHAP", gridcolor=cores['border']),
        yaxis=dict(title="", gridcolor=cores['border'], autorange="reversed"),
        showlegend=False,
        margin=dict(l=150, r=50, t=50, b=50)
    )
    return fig


def criar_grafico_shap_summary(shap_values, feature_names, titulo, cores):
    """Cria gráfico summary SHAP (beeswarm)"""
    if shap_values is None:
        return None
    
    n_features = min(10, len(feature_names))
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_shap)[-n_features:]
    top_features = [feature_names[i] for i in top_idx]
    
    fig = go.Figure()
    
    for i, feat in enumerate(reversed(top_features)):
        idx = feature_names.index(feat)
        values = shap_values[:, idx]
        
        y_pos = np.ones_like(values) * i + np.random.normal(0, 0.08, len(values))
        
        fig.add_trace(go.Scatter(
            x=values,
            y=y_pos,
            mode='markers',
            marker=dict(
                size=6,
                opacity=0.6,
                color=values,
                colorscale=[[0, cores['danger']], [0.5, cores['text_secondary']], [1, cores['success']]],
                showscale=False,
                line=dict(width=0)
            ),
            name=feat,
            hovertemplate='<b>%{text}</b><br>SHAP: %{x:.3f}<extra></extra>',
            text=[f"{data_manager.traduzir_coluna(feat)}" for _ in values]
        ))
    
    fig.update_layout(
        title=titulo,
        template='plotly_dark',
        paper_bgcolor=cores['card_bg'],
        plot_bgcolor=cores['card_bg'],
        font_color=cores['text'],
        height=350,
        xaxis=dict(title="SHAP Value", gridcolor=cores['border'], zeroline=True, zerolinecolor=cores['border']),
        yaxis=dict(
            title="",
            tickmode='array',
            tickvals=list(range(len(top_features))),
            ticktext=[data_manager.traduzir_coluna(f) for f in reversed(top_features)],
            gridcolor=cores['border'],
            showticklabels=True
        ),
        showlegend=False,
        hovermode='closest',
        margin=dict(l=20, r=20, t=50, b=50)
    )
    return fig


def criar_grafico_shap_waterfall(shap_values, feature_names, base_value, cores, modelo_tipo='classificacao'):
    """Cria gráfico waterfall SHAP para uma previsão individual"""
    if shap_values is None:
        return None
    
    sample_idx = np.random.randint(0, min(len(shap_values), 100))
    sample_shap = shap_values[sample_idx]
    
    idx_sorted = np.argsort(np.abs(sample_shap))[::-1][:8]
    
    feature_names_traduzidos = [data_manager.traduzir_coluna(f) for f in feature_names]
    
    waterfall_data = []
    cumulative = base_value if base_value is not None else 0
    
    for idx in idx_sorted:
        feat = feature_names_traduzidos[idx]
        value = sample_shap[idx]
        cumulative += value
        waterfall_data.append({
            'Feature': feat,
            'SHAP Value': value,
            'Cumulative': cumulative,
            'Color': 'positive' if value > 0 else 'negative'
        })
    
    df_waterfall = pd.DataFrame(waterfall_data)
    colors = [cores['success'] if v > 0 else cores['danger'] for v in df_waterfall['SHAP Value']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_waterfall['SHAP Value'],
        y=df_waterfall['Feature'],
        orientation='h',
        marker_color=colors,
        text=[f'{v:+.3f}' for v in df_waterfall['SHAP Value']],
        textposition='outside',
        textfont=dict(size=11),
        hovertemplate='<b>%{y}</b><br>SHAP: %{x:+.3f}<extra></extra>'
    ))
    
    fig.add_vline(x=0, line_dash="dash", line_color=cores['text_secondary'], line_width=1)
    
    fig.update_layout(
        title="SHAP Waterfall - Explicação da Previsão Individual",
        template='plotly_dark',
        paper_bgcolor=cores['card_bg'],
        plot_bgcolor=cores['card_bg'],
        font_color=cores['text'],
        height=350,
        xaxis=dict(title="SHAP Value", gridcolor=cores['border'], zeroline=True),
        yaxis=dict(title="", gridcolor=cores['border'], autorange="reversed"),
        showlegend=False,
        margin=dict(l=150, r=50, t=50, b=50)
    )
    
    return fig

# ==================================================
# LAYOUT
# ==================================================

def create_layout(df):
    """Layout da página de Comparação de Modelos com XAI"""
    
    modelos = model_manager.carregar_modelos_comparativos()
    tem_modelos = modelos['classificacao']['model'] is not None and modelos['regressao']['model'] is not None
    
    return html.Div([
        html.Div([
            dbc.Button("← Voltar", href="/", color="light", size="sm",
                      style={'backgroundColor': 'transparent', 'border': f'1px solid {CORES["border"]}', 
                            'color': CORES['text']})
        ], style={'position': 'fixed', 'top': '20px', 'left': '20px', 'zIndex': '1000'}),
        
        html.Div([
            html.H3("🔍 Comparação de Modelos com XAI", style={'color': CORES['text'], 'marginBottom': '10px'}),
            html.P("Compare as previsões e entenda o impacto de cada feature com SHAP",
                  style={'color': CORES['text_secondary'], 'marginBottom': '30px'}),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("📊 Classificação", style={'color': CORES['accent']}),
                            html.P("Status: " + ("✅ Disponível" if tem_modelos else "❌ Não disponível"),
                                  style={'color': CORES['success'] if tem_modelos else CORES['danger']}),
                            html.P("🔮 SHAP: " + ("✅ Disponível" if SHAP_AVAILABLE else "❌ Não instalado"),
                                  style={'color': CORES['success'] if SHAP_AVAILABLE else CORES['danger']}),
                        ])
                    ], style={'backgroundColor': CORES['card_bg'], 'border': f'1px solid {CORES["border"]}'})
                ], md=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("📈 Regressão", style={'color': CORES['success']}),
                            html.P("Status: " + ("✅ Disponível" if tem_modelos else "❌ Não disponível"),
                                  style={'color': CORES['success'] if tem_modelos else CORES['danger']}),
                            html.P("🔮 SHAP: " + ("✅ Disponível" if SHAP_AVAILABLE else "❌ Não instalado"),
                                  style={'color': CORES['success'] if SHAP_AVAILABLE else CORES['danger']}),
                        ])
                    ], style={'backgroundColor': CORES['card_bg'], 'border': f'1px solid {CORES["border"]}'})
                ], md=6),
            ], className="mb-4"),
            
            html.Div(id='comparacao-content')
            
        ], style={'maxWidth': '1400px', 'margin': '0 auto', 'padding': '80px 20px 40px 20px'})
        
    ], style={'backgroundColor': CORES['background'], 'minHeight': '100vh', 'padding': '20px'})


# ================================================
# CALLBACK PARA CARREGAR CONTEÚDO
# ================================================

@callback(
    Output('comparacao-content', 'children'),
    Input('comparacao-content', 'id')
)
def carregar_conteudo_comparacao(_):
    """Carrega o conteúdo da página de comparação"""
    
    df = data_manager.get_clean_df()
    if df is None:
        return html.Div([
            html.H4("❌ Dados não carregados", style={'color': CORES['danger']})
        ])
    
    features_disponiveis = [f for f in TODAS_FEATURES if f in df.columns]
    if not features_disponiveis:
        features_disponiveis = FEATURES_PADRAO
    
    modelos = model_manager.carregar_modelos_comparativos()
    tem_modelos = modelos['classificacao']['model'] is not None and modelos['regressao']['model'] is not None
    
    if not tem_modelos:
        return html.Div([
            html.Div([
                html.H4("⚠️ Modelos não disponíveis", style={'color': CORES['warning']}),
                html.P("Treine e salve os modelos na página de Classificação.",
                      style={'color': CORES['text_secondary']}),
                dbc.Button("Ir para Classificação", href="/classificacao", color="primary",
                          style={'backgroundColor': CORES['accent'], 'border': 'none', 'marginTop': '20px'})
            ], style={'textAlign': 'center', 'padding': '60px 20px'})
        ])
    
    features = modelos['classificacao']['features']
    features_validas = [f for f in features if f in df.columns]
    if not features_validas:
        features_validas = features_disponiveis[:5]
    
    inputs = []
    for feature in features_validas:
        nome_traduzido = data_manager.traduzir_coluna(feature)
        is_derivada = feature in FEATURES_DERIVADAS
        
        label_text = nome_traduzido + " 📐" if is_derivada else nome_traduzido
        
        inputs.append(html.Div([
            html.Label(label_text, style={'color': CORES['text'], 'fontSize': '14px', 'marginBottom': '5px'}),
            dbc.Input(
                type="number",
                id=f"comp-input-{feature}",
                value=None,
                step=0.01,
                placeholder=f"Digite o valor",
                style={
                    'backgroundColor': CORES['card_bg'], 
                    'color': CORES['text'], 
                    'border': f'1px solid {CORES["border"]}',
                    'width': '100%'
                }
            )
        ], style={'marginBottom': '15px'}))
    
    info_shap = html.Div([
        html.P(
            "🔮 SHAP disponível para explicar as previsões!" if SHAP_AVAILABLE else "⚠️ SHAP não instalado. Instale com: pip install shap",
            style={'color': CORES['success'] if SHAP_AVAILABLE else CORES['warning'], 'fontSize': '12px', 'marginTop': '10px'}
        )
    ])
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H4("📝 Insira seus dados", style={'color': CORES['text'], 'marginBottom': '20px'}),
                html.Div(inputs),
                info_shap,
                dbc.Button(
                    "🔍 Comparar Modelos com XAI",
                    id="compare-models-button",
                    color="primary",
                    size="lg",
                    className="mt-3 w-100",
                    style={'backgroundColor': CORES['accent'], 'border': 'none'}
                ),
            ], md=4),
            
            dbc.Col([
                html.H4("📊 Resultados e Explicações (SHAP)", style={'color': CORES['text'], 'marginBottom': '20px'}),
                html.Div(id='comparacao-resultados', children=[
                    html.P("Preencha os dados e clique em 'Comparar Modelos com XAI'",
                          style={'color': CORES['text_secondary'], 'textAlign': 'center', 'marginTop': '50px'})
                ])
            ], md=8)
        ])
    ], style={'marginTop': '20px'})


# ================================================
# CALLBACK DA COMPARAÇÃO COM XAI
# ================================================

@callback(
    Output('comparacao-resultados', 'children'),
    Input('compare-models-button', 'n_clicks'),
    [State(f'comp-input-{f}', 'value') for f in TODAS_FEATURES],
    prevent_initial_call=True
)
def comparar_modelos_com_xai(n_clicks, hrv, resting_heart_rate, day_strain, sleep_hours, sleep_efficiency,
                             sleep_quality, strain_per_sleep, hrv_ratio, hrv_rhr_ratio):
    """Compara os modelos com explicabilidade SHAP"""
    
    if n_clicks is None:
        return html.P("Preencha os dados e clique em 'Comparar Modelos com XAI'",
                     style={'color': CORES['text_secondary'], 'textAlign': 'center', 'marginTop': '30px'})
    
    modelos = model_manager.carregar_modelos_comparativos()
    
    if modelos['classificacao']['model'] is None or modelos['regressao']['model'] is None:
        return html.Div([
            html.H4("❌ Modelos não disponíveis", style={'color': CORES['danger']}),
            html.P("Treine ambos os modelos na página de Classificação.",
                  style={'color': CORES['text_secondary']})
        ])
    
    # Mapear valores
    valores_input = {
        'hrv': hrv,
        'resting_heart_rate': resting_heart_rate,
        'day_strain': day_strain,
        'sleep_hours': sleep_hours,
        'sleep_efficiency': sleep_efficiency,
        'sleep_quality': sleep_quality,
        'strain_per_sleep': strain_per_sleep,
        'hrv_ratio': hrv_ratio,
        'hrv_rhr_ratio': hrv_rhr_ratio
    }
    
    features = modelos['classificacao']['features']
    
    dados = {}
    campos_vazios = []
    for feature in features:
        valor = valores_input.get(feature)
        if valor is None or valor == '':
            campos_vazios.append(feature)
        else:
            dados[feature] = float(valor)
    
    if campos_vazios:
        nomes = [data_manager.traduzir_coluna(f) for f in campos_vazios]
        return html.Div([
            html.H4("⚠️ Campos incompletos", style={'color': CORES['warning']}),
            html.P(f"Preencha: {', '.join(nomes)}", style={'color': CORES['text_secondary']})
        ])
    
    try:
        df_usuario = pd.DataFrame([dados])
        X = df_usuario[features].copy()
        
        # ================================================
        # PREVISÃO DA CLASSIFICAÇÃO
        # ================================================
        clf_model = modelos['classificacao']['model']
        clf_scaler = modelos['classificacao']['scaler']
        
        if clf_scaler is not None:
            X_scaled = clf_scaler.transform(X)
            X_clf = pd.DataFrame(X_scaled, columns=features)
        else:
            X_clf = X
        
        if hasattr(clf_model, 'predict_proba'):
            proba = clf_model.predict_proba(X_clf)[0, 1]
            score_clf = proba * 100
        else:
            score_clf = clf_model.predict(X_clf)[0] * 100
        
        # ================================================
        # PREVISÃO DA REGRESSÃO
        # ================================================
        reg_model = modelos['regressao']['model']
        reg_scaler = modelos['regressao']['scaler']
        
        if reg_scaler is not None:
            X_scaled = reg_scaler.transform(X)
            X_reg = pd.DataFrame(X_scaled, columns=features)
        else:
            X_reg = X
        
        score_reg = reg_model.predict(X_reg)[0]
        
        # ================================================
        # SHAP - CLASSIFICAÇÃO
        # ================================================
        shap_clf_values = None
        shap_clf_importance = None
        shap_clf_summary = None
        shap_clf_waterfall = None
        
        if SHAP_AVAILABLE:
            try:
                X_shap = X.copy()
                
                shap_clf_values, explainer_clf = calcular_shap_para_modelo(
                    clf_model, X_shap, features, 'classificacao'
                )
                
                if shap_clf_values is not None:
                    shap_clf_importance = criar_grafico_shap_importance(
                        shap_clf_values, features, "SHAP - Classificação", CORES
                    )
                    shap_clf_summary = criar_grafico_shap_summary(
                        shap_clf_values, features, "SHAP Summary - Classificação", CORES
                    )
                    base_value = explainer_clf.expected_value if explainer_clf is not None else None
                    if isinstance(base_value, list):
                        base_value = base_value[0] if len(base_value) >= 2 else base_value[0]
                    shap_clf_waterfall = criar_grafico_shap_waterfall(
                        shap_clf_values, features, base_value, CORES, 'classificacao'
                    )
            except Exception as e:
                print(f"Erro SHAP classificação: {e}")
        
        # ================================================
        # SHAP - REGRESSÃO
        # ================================================
        shap_reg_values = None
        shap_reg_importance = None
        shap_reg_summary = None
        shap_reg_waterfall = None
        
        if SHAP_AVAILABLE:
            try:
                X_shap = X.copy()
                
                shap_reg_values, explainer_reg = calcular_shap_para_modelo(
                    reg_model, X_shap, features, 'regressao'
                )
                
                if shap_reg_values is not None:
                    shap_reg_importance = criar_grafico_shap_importance(
                        shap_reg_values, features, "SHAP - Regressão", CORES
                    )
                    shap_reg_summary = criar_grafico_shap_summary(
                        shap_reg_values, features, "SHAP Summary - Regressão", CORES
                    )
                    base_value = explainer_reg.expected_value if explainer_reg is not None else 0
                    shap_reg_waterfall = criar_grafico_shap_waterfall(
                        shap_reg_values, features, base_value, CORES, 'regressao'
                    )
            except Exception as e:
                print(f"Erro SHAP regressão: {e}")
        
        # ================================================
        # CÁLCULO DAS DIFERENÇAS
        # ================================================
        diferenca = score_reg - score_clf
        diferenca_pct = (diferenca / score_clf) * 100 if score_clf > 0 else 0
        
        # ================================================
        # STATUS
        # ================================================
        def get_status(score):
            if score >= 80:
                return {'text': '🟢 Alta Recuperação', 'cor': CORES['success']}
            elif score >= 60:
                return {'text': '🟡 Recuperação Moderada', 'cor': CORES['warning']}
            else:
                return {'text': '🔴 Baixa Recuperação', 'cor': CORES['danger']}
        
        status_clf = get_status(score_clf)
        status_reg = get_status(score_reg)
        
        # ================================================
        # GRÁFICO DE COMPARAÇÃO
        # ================================================
        fig_comparison = go.Figure()
        fig_comparison.add_trace(go.Bar(
            x=['Classificação', 'Regressão'],
            y=[score_clf, score_reg],
            text=[f'{score_clf:.1f}', f'{score_reg:.1f}'],
            textposition='auto',
            marker_color=[CORES['accent'], CORES['success']],
            name='Score'
        ))
        fig_comparison.add_hline(y=75, line_dash="dash", line_color=CORES['text_secondary'], 
                                 annotation_text="Alta (75)")
        fig_comparison.add_hline(y=50, line_dash="dash", line_color=CORES['text_secondary'],
                                 annotation_text="Moderada (50)")
        fig_comparison.update_layout(
            title="Comparação dos Scores",
            template='plotly_dark',
            paper_bgcolor=CORES['card_bg'],
            plot_bgcolor=CORES['card_bg'],
            font_color=CORES['text'],
            height=300,
            yaxis=dict(title="Score", range=[0, 100], gridcolor=CORES['border']),
            xaxis=dict(title="Modelo", gridcolor=CORES['border']),
            showlegend=False
        )
        
        # ================================================
        # TABELA DE COMPARAÇÃO
        # ================================================
        tabela_comparacao = dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("Métrica", style={'color': CORES['text']}),
                    html.Th("Classificação", style={'color': CORES['accent']}),
                    html.Th("Regressão", style={'color': CORES['success']}),
                    html.Th("Diferença", style={'color': CORES['warning']})
                ])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td("Score", style={'color': CORES['text_secondary']}),
                    html.Td(f"{score_clf:.1f}", style={'color': status_clf['cor']}),
                    html.Td(f"{score_reg:.1f}", style={'color': status_reg['cor']}),
                    html.Td(f"{diferenca:+.1f}", style={'color': CORES['warning']})
                ]),
                html.Tr([
                    html.Td("Status", style={'color': CORES['text_secondary']}),
                    html.Td(status_clf['text'], style={'color': status_clf['cor']}),
                    html.Td(status_reg['text'], style={'color': status_reg['cor']}),
                    html.Td("—", style={'color': CORES['text_secondary']})
                ]),
                html.Tr([
                    html.Td("Diferença %", style={'color': CORES['text_secondary']}),
                    html.Td("—", style={'color': CORES['text_secondary']}),
                    html.Td("—", style={'color': CORES['text_secondary']}),
                    html.Td(f"{diferenca_pct:+.1f}%", style={'color': CORES['warning']})
                ])
            ])
        ], bordered=True, hover=True, striped=True, style={'color': CORES['text']})
        
        # ================================================
        # LAYOUT FINAL - GRÁFICOS EM LINHAS SEPARADAS
        # ================================================
        
        # Inicializar layout
        layout_final = html.Div([
            # Cards lado a lado
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("📊 Classificação", style={'color': CORES['accent']}),
                            html.H2(f"{score_clf:.1f}", style={'color': status_clf['cor']}),
                            html.H4(status_clf['text'], style={'color': status_clf['cor']}),
                            html.P(f"Probabilidade: {proba:.1%}", style={'color': CORES['text_secondary']}),
                        ])
                    ], style={'backgroundColor': CORES['card_bg'], 'border': f'3px solid {status_clf["cor"]}'})
                ], md=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("📈 Regressão", style={'color': CORES['success']}),
                            html.H2(f"{score_reg:.1f}", style={'color': status_reg['cor']}),
                            html.H4(status_reg['text'], style={'color': status_reg['cor']}),
                            html.P(f"Score contínuo", style={'color': CORES['text_secondary']}),
                        ])
                    ], style={'backgroundColor': CORES['card_bg'], 'border': f'3px solid {status_reg["cor"]}'})
                ], md=6),
            ], className="mb-4"),
            
            # Diferença
            dbc.Card([
                dbc.CardBody([
                    html.H5("📊 Diferença entre os Modelos", style={'color': CORES['text']}),
                    html.H2(
                        f"{diferenca:+.1f} pontos ({diferenca_pct:+.1f}%)",
                        style={'color': CORES['warning'] if abs(diferenca) < 10 else CORES['danger']}
                    ),
                    html.P(
                        "✅ Modelos alinhados!" if abs(diferenca) < 10 else "⚠️ Modelos divergem.",
                        style={'color': CORES['text_secondary']}
                    )
                ])
            ], style={'backgroundColor': CORES['card_bg'], 'border': f'1px solid {CORES["border"]}', 'marginBottom': '20px'}),
            
            # Tabela
            tabela_comparacao,
            
            # ================================================
            # GRÁFICO DE COMPARAÇÃO - LINHA COMPLETA
            # ================================================
            html.Hr(style={'borderColor': CORES['border'], 'marginTop': '30px'}),
            html.H4("📊 Comparação Visual dos Scores", style={'color': CORES['text']}),
            html.P("Comparação entre os scores dos dois modelos",
                  style={'color': CORES['text_secondary'], 'fontSize': '14px', 'marginBottom': '20px'}),
            dcc.Graph(figure=fig_comparison, config={'displayModeBar': False}),
            
            # ================================================
            # SEÇÃO XAI - SHAP
            # ================================================
            html.Hr(style={'borderColor': CORES['border'], 'marginTop': '30px'}),
            html.H4("🔮 Explicação das Previsões (SHAP)", style={'color': CORES['text']}),
            html.P("Entenda como cada feature contribui para a previsão de cada modelo",
                  style={'color': CORES['text_secondary'], 'fontSize': '14px', 'marginBottom': '20px'}),
        ])
        
        # ================================================
        # SHAP - CLASSIFICAÇÃO (GRÁFICOS EM LINHAS SEPARADAS)
        # ================================================
        if shap_clf_importance is not None:
            layout_final.children += (
                html.H5("📊 SHAP - Classificação", style={'color': CORES['accent'], 'marginTop': '20px'}),
                
                # Importância - LINHA COMPLETA
                html.P("Importância das Features (SHAP)", style={'color': CORES['text_secondary'], 'fontSize': '12px'}),
                dcc.Graph(figure=shap_clf_importance, config={'displayModeBar': False}),
                
                # Summary Plot - LINHA COMPLETA
                html.P("Distribuição do Impacto (SHAP Summary)", style={'color': CORES['text_secondary'], 'fontSize': '12px', 'marginTop': '20px'}),
                dcc.Graph(figure=shap_clf_summary, config={'displayModeBar': False}) if shap_clf_summary else None,
                
                # Waterfall - LINHA COMPLETA
                html.P("Exemplo de Previsão Individual (SHAP Waterfall)", style={'color': CORES['text_secondary'], 'fontSize': '12px', 'marginTop': '20px'}),
                dcc.Graph(figure=shap_clf_waterfall, config={'displayModeBar': False}) if shap_clf_waterfall else None,
            )
        else:
            layout_final.children += (
                html.H5("📊 SHAP - Classificação", style={'color': CORES['accent'], 'marginTop': '20px'}),
                html.P("SHAP não disponível para o modelo de classificação", style={'color': CORES['warning']}),
            )
        
        # ================================================
        # SHAP - REGRESSÃO (GRÁFICOS EM LINHAS SEPARADAS)
        # ================================================
        if shap_reg_importance is not None:
            layout_final.children += (
                html.Hr(style={'borderColor': CORES['border'], 'marginTop': '30px'}),
                html.H5("📈 SHAP - Regressão", style={'color': CORES['success'], 'marginTop': '20px'}),
                
                # Importância - LINHA COMPLETA
                html.P("Importância das Features (SHAP)", style={'color': CORES['text_secondary'], 'fontSize': '12px'}),
                dcc.Graph(figure=shap_reg_importance, config={'displayModeBar': False}),
                
                # Summary Plot - LINHA COMPLETA
                html.P("Distribuição do Impacto (SHAP Summary)", style={'color': CORES['text_secondary'], 'fontSize': '12px', 'marginTop': '20px'}),
                dcc.Graph(figure=shap_reg_summary, config={'displayModeBar': False}) if shap_reg_summary else None,
                
                # Waterfall - LINHA COMPLETA
                html.P("Exemplo de Previsão Individual (SHAP Waterfall)", style={'color': CORES['text_secondary'], 'fontSize': '12px', 'marginTop': '20px'}),
                dcc.Graph(figure=shap_reg_waterfall, config={'displayModeBar': False}) if shap_reg_waterfall else None,
            )
        else:
            layout_final.children += (
                html.Hr(style={'borderColor': CORES['border'], 'marginTop': '30px'}),
                html.H5("📈 SHAP - Regressão", style={'color': CORES['success'], 'marginTop': '20px'}),
                html.P("SHAP não disponível para o modelo de regressão", style={'color': CORES['warning']}),
            )
        
        # ================================================
        # RODAPÉ E VALORES
        # ================================================
        layout_final.children += (
            # Legenda SHAP
            html.Hr(style={'borderColor': CORES['border'], 'marginTop': '30px'}),
            html.Div([
                html.P(
                    "🔵 SHAP: Valores positivos (azul) indicam que a feature contribui para aumentar o score. "
                    "🔴 Valores negativos (vermelho) indicam que a feature contribui para diminuir o score.",
                    style={'color': CORES['text_secondary'], 'fontSize': '12px', 'textAlign': 'center'}
                )
            ]),
            
            # Valores analisados
            html.Hr(style={'borderColor': CORES['border']}),
            html.H5("📋 Valores Analisados", style={'color': CORES['text']}),
            html.Div([
                html.Div([
                    html.Div([
                        html.Span(data_manager.traduzir_coluna(feature), style={'color': CORES['text_secondary']}),
                        html.Span(f"{dados.get(feature, 0):.2f}", 
                                  style={'color': CORES['text'], 'float': 'right', 'fontWeight': 'bold'})
                    ], style={'padding': '5px 0', 'borderBottom': f'1px solid {CORES["border"]}'})
                ]) for feature in features
            ]),
            
            # Recomendação
            html.Hr(style={'borderColor': CORES['border']}),
            html.Div([
                html.H5("💡 Recomendação", style={'color': CORES['text']}),
                html.P(
                    "Use a Classificação para decisões binárias e a Regressão para acompanhamento contínuo. "
                    "O SHAP ajuda a entender quais fatores estão influenciando sua recuperação.",
                    style={'color': CORES['text_secondary']}
                )
            ]),
            
            # Rodapé
            html.Hr(style={'borderColor': CORES['border']}),
            html.P(
                f"📊 {len(features)} features | SHAP: {'✅ Disponível' if SHAP_AVAILABLE else '❌ Não instalado'}",
                style={'color': CORES['text_secondary'], 'fontSize': '11px', 'textAlign': 'center'}
            )
        )
        
        return layout_final
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return html.Div([
            html.H4("❌ Erro na análise", style={'color': CORES['danger']}),
            html.P(f"Erro: {str(e)}", style={'color': CORES['text_secondary']})
        ])