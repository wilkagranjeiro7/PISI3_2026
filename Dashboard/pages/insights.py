# ==================================================
# pages/insights.py - ANÁLISE DE INSIGHTS (COMPLETA)
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

# ==================================================
# CORES
# ==================================================

CORES = data_manager.get_cores()

# ==================================================
# LAYOUT
# ==================================================

def create_layout(df):
    """Layout da página de Insights"""
    
    # Carregar modelo salvo
    model, scaler, features, metadata = model_manager.carregar_melhor_modelo()
    
    if model is None:
        return html.Div([
            html.Div([
                dbc.Button("← Voltar", href="/", color="light", size="sm",
                          style={'backgroundColor': 'transparent', 'border': f'1px solid {CORES["border"]}', 
                                'color': CORES['text']})
            ], style={'position': 'fixed', 'top': '20px', 'left': '20px', 'zIndex': '1000'}),
            
            html.Div([
                html.H3("Análise de Insights", style={'color': CORES['text'], 'marginBottom': '30px'}),
                html.Div([
                    html.H4("Nenhum modelo disponível para análise", style={'color': CORES['warning']}),
                    html.P("Treine e salve um modelo na página de Classificação primeiro.",
                          style={'color': CORES['text_secondary']}),
                    dbc.Button("Ir para Classificação", href="/classificacao", color="primary",
                              style={'backgroundColor': CORES['accent'], 'border': 'none', 'marginTop': '20px'})
                ], style={'textAlign': 'center', 'padding': '60px 20px'})
            ], style={'maxWidth': '800px', 'margin': '0 auto', 'padding': '40px 20px'})
        ], style={'backgroundColor': CORES['background'], 'minHeight': '100vh', 'padding': '20px'})
    
    # Criar inputs para cada feature
    inputs = []
    for feature in features:
        nome_traduzido = data_manager.traduzir_coluna(feature)
        valor_default = float(df[feature].mean()) if feature in df.columns else 0
        min_val = float(df[feature].min()) if feature in df.columns else 0
        max_val = float(df[feature].max()) if feature in df.columns else 100
        
        inputs.append(html.Div([
            html.Label(nome_traduzido, style={'color': CORES['text'], 'fontSize': '14px', 'marginBottom': '5px'}),
            html.Div([
                dbc.Input(
                    type="number",
                    id=f"input-{feature}",
                    value=valor_default,
                    step=0.01,
                    style={
                        'backgroundColor': CORES['card_bg'], 
                        'color': CORES['text'], 
                        'border': f'1px solid {CORES["border"]}',
                        'width': '100%'
                    }
                ),
                html.Small(
                    f"Faixa típica: {min_val:.1f} - {max_val:.1f}",
                    style={'color': CORES['text_secondary'], 'fontSize': '11px', 'display': 'block', 'marginTop': '3px'}
                )
            ])
        ], style={'marginBottom': '15px'}))
    
    return html.Div([
        html.Div([
            dbc.Button("← Voltar", href="/", color="light", size="sm",
                      style={'backgroundColor': 'transparent', 'border': f'1px solid {CORES["border"]}', 
                            'color': CORES['text']})
        ], style={'position': 'fixed', 'top': '20px', 'left': '20px', 'zIndex': '1000'}),
        
        html.Div([
            html.H3("Análise de Insights para Recuperação", style={'color': CORES['text'], 'marginBottom': '10px'}),
            html.P("Entenda como os fatores influenciam a recuperação e receba recomendações personalizadas",
                  style={'color': CORES['text_secondary'], 'marginBottom': '30px'}),
            
            # Info do modelo
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Modelo Ativo", style={'color': CORES['accent']}),
                            html.P(f"{metadata.get('nome', 'Desconhecido')}", 
                                  style={'color': CORES['text'], 'fontWeight': 'bold'}),
                        ], md=4),
                        dbc.Col([
                            html.H5("Acurácia", style={'color': CORES['text_secondary'], 'fontSize': '12px'}),
                            html.P(f"{metadata.get('metricas', {}).get('accuracy', 0):.1%}", 
                                  style={'color': CORES['success'], 'fontSize': '20px'}),
                        ], md=3),
                        dbc.Col([
                            html.H5("F1-Score", style={'color': CORES['text_secondary'], 'fontSize': '12px'}),
                            html.P(f"{metadata.get('metricas', {}).get('f1_score', 0):.3f}", 
                                  style={'color': CORES['accent'], 'fontSize': '20px'}),
                        ], md=3),
                        dbc.Col([
                            html.H5("Análise em", style={'color': CORES['text_secondary'], 'fontSize': '12px'}),
                            html.P(f"{metadata.get('data_criacao', '')[:10]}", 
                                  style={'color': CORES['text_secondary'], 'fontSize': '14px'}),
                        ], md=2),
                    ])
                ])
            ], style={'backgroundColor': CORES['card_bg'], 'border': f'1px solid {CORES["border"]}',
                     'marginBottom': '30px'}),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Informe os Dados Atuais", style={'color': CORES['text'], 'marginBottom': '20px'}),
                    html.Div(inputs),
                    
                    dbc.Button(
                        "Analisar Recuperação",
                        id="analyze-button",
                        color="primary",
                        size="lg",
                        className="mt-3",
                        style={'backgroundColor': CORES['accent'], 'border': 'none', 'width': '100%'}
                    ),
                ], md=5),
                
                dbc.Col([
                    html.H4("Resultado da Análise", style={'color': CORES['text'], 'marginBottom': '20px'}),
                    html.Div(id='analysis-result', style={'minHeight': '300px'})
                ], md=7)
            ])
            
        ], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '80px 20px 40px 20px'})
        
    ], style={'backgroundColor': CORES['background'], 'minHeight': '100vh', 'padding': '20px'})


# ==================================================
# FUNÇÕES AUXILIARES
# ==================================================

def determinar_nivel_recuperacao(probabilidade, score_previsto):
    """
    Determina o nível de recuperação baseado na probabilidade e no score
    """
    if score_previsto >= 80:
        return {
            'nivel': 'Alta Recuperação',
            'cor': '#10B981',
            'icone': '🟢',
            'mensagem': 'Excelente! Seu corpo está totalmente recuperado. Aproveite para treinar com alta intensidade!',
            'recomendacao': 'Treino intenso recomendado',
            'status': 'excelente'
        }
    elif score_previsto >= 60:
        return {
            'nivel': 'Recuperação Moderada',
            'cor': '#F59E0B',
            'icone': '🟡',
            'mensagem': 'Recuperação razoável. Você pode treinar, mas com intensidade moderada.',
            'recomendacao': 'Treino moderado recomendado',
            'status': 'moderado'
        }
    else:
        return {
            'nivel': 'Baixa Recuperação',
            'cor': '#EF4444',
            'icone': '🔴',
            'mensagem': 'Seu corpo precisa de descanso. Priorize recuperação e evite treinos intensos.',
            'recomendacao': 'Descanso ou treino leve recomendado',
            'status': 'baixo'
        }


def analisar_feature_importance(feature_importance, features, dados_atuais, df_referencia):
    """
    Analisa quais features mais contribuíram para o resultado
    """
    if not feature_importance or len(feature_importance) != len(features):
        return None
    
    # Criar DataFrame com importâncias
    df_importancia = pd.DataFrame({
        'feature': features,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    # Para cada feature, comparar com a média
    insights = []
    for _, row in df_importancia.iterrows():
        feature = row['feature']
        importance = row['importance']
        valor_atual = dados_atuais.get(feature, 0)
        
        # Média da feature no dataset
        media = df_referencia[feature].mean() if feature in df_referencia.columns else 0
        
        # Percentual de contribuição
        contribuicao = importance * (valor_atual - media)
        
        insights.append({
            'feature': feature,
            'importance': importance,
            'valor_atual': valor_atual,
            'media': media,
            'contribuicao': contribuicao,
            'impacto': 'positivo' if contribuicao > 0 else 'negativo'
        })
    
    return insights


def calcular_tendencia(dados_atuais, df_referencia, features):
    """
    Calcula a tendência baseada nos valores atuais em comparação com o histórico
    """
    tendencias = []
    for feature in features:
        if feature in dados_atuais and feature in df_referencia.columns:
            valor_atual = dados_atuais[feature]
            media = df_referencia[feature].mean()
            desvio_padrao = df_referencia[feature].std()
            
            if desvio_padrao > 0:
                z_score = (valor_atual - media) / desvio_padrao
                if z_score > 0.5:
                    tendencias.append(('📈', feature, 'acima da média'))
                elif z_score < -0.5:
                    tendencias.append(('📉', feature, 'abaixo da média'))
                else:
                    tendencias.append(('➡️', feature, 'na média'))
    
    return tendencias


# ==================================================
# CALLBACK DA ANÁLISE
# ==================================================

@callback(
    Output('analysis-result', 'children'),
    Input('analyze-button', 'n_clicks'),
    [State(f'input-{f}', 'value') for f in ['hrv', 'resting_heart_rate', 'day_strain', 'sleep_hours', 'sleep_efficiency', 'sleep_quality', 'strain_per_sleep', 'hrv_rhr_ratio', 'hrv_ratio']],
    prevent_initial_call=True
)
def fazer_analise(n_clicks, hrv, resting_heart_rate, day_strain, sleep_hours, sleep_efficiency, sleep_quality, strain_per_sleep, hrv_rhr_ratio, hrv_ratio):
    """Faz a análise com os dados inseridos usando o modelo salvo"""
    
    if n_clicks is None:
        return html.Div(
            html.P("Preencha os dados e clique em 'Analisar Recuperação' para obter insights",
                  style={'color': CORES['text_secondary'], 'textAlign': 'center', 'marginTop': '50px'})
        )
    
    # Carregar modelo salvo
    model, scaler, features, metadata = model_manager.carregar_melhor_modelo()
    
    if model is None:
        return html.Div([
            html.H4("Erro: Nenhum modelo disponível", style={'color': CORES['danger']}),
            html.P("Treine e salve um modelo na página de Classificação primeiro.",
                  style={'color': CORES['text_secondary']})
        ])
    
    try:
        # Construir dicionário com os dados dos States
        valores_input = {
            'hrv': hrv,
            'resting_heart_rate': resting_heart_rate,
            'day_strain': day_strain,
            'sleep_hours': sleep_hours,
            'sleep_efficiency': sleep_efficiency,
            'sleep_quality': sleep_quality,
            'strain_per_sleep': strain_per_sleep,
            'hrv_rhr_ratio': hrv_rhr_ratio,
            'hrv_ratio': hrv_ratio
        }
        
        # Pegar apenas as features que o modelo usa
        dados = {}
        for feature in features:
            if feature in valores_input and valores_input[feature] is not None:
                dados[feature] = float(valores_input[feature])
            else:
                dados[feature] = 0.0
        
        # Criar DataFrame com os dados
        df_usuario = pd.DataFrame([dados])
        
        # Garantir que tem todas as features na ordem correta
        X = df_usuario[features].copy()
        
        # Normalizar se necessário
        if scaler is not None:
            X_scaled = scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=features)
        
        # Fazer previsão
        if hasattr(model, 'predict_proba'):
            proba_alta = model.predict_proba(X)[0, 1]
            predicao = 1 if proba_alta >= 0.5 else 0
        else:
            predicao = model.predict(X)[0]
            proba_alta = predicao
        
        # Score de recuperação
        score_recuperacao = proba_alta * 100
        
        # Confiança
        if hasattr(model, 'predict_proba'):
            confianca = proba_alta if predicao == 1 else 1 - proba_alta
        else:
            confianca = 1.0
        
        # Determinar o nível de recuperação
        nivel = determinar_nivel_recuperacao(proba_alta, score_recuperacao)
        
        # Carregar DataFrame de referência para comparações
        df_referencia = data_manager.get_clean_df()
        
        # ============================================
        # ANÁLISE DE FEATURE IMPORTANCE (OPÇÃO 4)
        # ============================================
        feature_importance = metadata.get('metricas', {}).get('feature_importance', [])
        insights_features = None
        
        if feature_importance and len(feature_importance) == len(features):
            insights_features = analisar_feature_importance(
                feature_importance, features, dados, df_referencia
            )
        
        # ============================================
        # TENDÊNCIA (OPÇÃO 4)
        # ============================================
        tendencias = calcular_tendencia(dados, df_referencia, features)
        
        # ============================================
        # MONTAR RESULTADO
        # ============================================
        cor = nivel['cor']
        icone = nivel['icone']
        
        # Componentes do resultado
        resultado_analise = [
            # Status Principal
            html.Div([
                html.H4(f"{icone} Status da Recuperação", style={'color': CORES['text']}),
                html.H2(
                    nivel['nivel'],
                    style={'color': cor, 'marginTop': '15px', 'fontWeight': 'bold'}
                ),
            ]),
            
            # Score de recuperação
            html.Div([
                html.P(
                    f"Score de Recuperação: {score_recuperacao:.1f} / 100",
                    style={'color': CORES['text_secondary'], 'fontSize': '18px', 'marginTop': '10px'}
                ),
                html.Div([
                    html.Div(
                        style={
                            'width': f"{score_recuperacao}%",
                            'height': '30px',
                            'backgroundColor': cor,
                            'borderRadius': '15px',
                            'transition': 'width 0.5s',
                            'display': 'flex',
                            'alignItems': 'center',
                            'justifyContent': 'center',
                            'color': 'white',
                            'fontWeight': 'bold'
                        }
                    )
                ], style={
                    'width': '100%',
                    'height': '30px',
                    'backgroundColor': CORES['card_bg'],
                    'borderRadius': '15px',
                    'marginTop': '10px',
                    'overflow': 'hidden',
                    'border': f'1px solid {CORES["border"]}'
                })
            ]),
            
            html.P(
                f"Probabilidade de Alta Recuperação: {proba_alta:.1%}",
                style={'color': CORES['text_secondary'], 'fontSize': '14px', 'marginTop': '10px'}
            ),
            
            html.Hr(style={'borderColor': CORES['border']}),
        ]
        
        # ============================================
        # INSIGHTS - FEATURES MAIS IMPORTANTES
        # ============================================
        if insights_features:
            # Separar features positivas e negativas
            positivas = [i for i in insights_features if i['impacto'] == 'positivo']
            negativas = [i for i in insights_features if i['impacto'] == 'negativo']
            
            resultado_analise.extend([
                html.H5("🔍 Fatores que mais influenciaram", style={'color': CORES['text']}),
            ])
            
            if positivas:
                # Pegar a feature mais positiva
                top_positive = max(positivas, key=lambda x: abs(x['contribuicao']))
                nome_feature = data_manager.traduzir_coluna(top_positive['feature'])
                resultado_analise.append(
                    html.Div([
                        html.Span("🟢 ", style={'color': CORES['success']}),
                        html.Span(f"Contribuição positiva: ", style={'color': CORES['text_secondary']}),
                        html.Span(f"{nome_feature} ", style={'color': CORES['success'], 'fontWeight': 'bold'}),
                        html.Span(f"(valor: {top_positive['valor_atual']:.2f} vs média: {top_positive['media']:.2f})",
                                 style={'color': CORES['text_secondary'], 'fontSize': '12px'})
                    ], style={'marginBottom': '5px'})
                )
            
            if negativas:
                # Pegar a feature mais negativa
                top_negative = max(negativas, key=lambda x: abs(x['contribuicao']))
                nome_feature = data_manager.traduzir_coluna(top_negative['feature'])
                resultado_analise.append(
                    html.Div([
                        html.Span("🔴 ", style={'color': CORES['danger']}),
                        html.Span(f"Contribuição negativa: ", style={'color': CORES['text_secondary']}),
                        html.Span(f"{nome_feature} ", style={'color': CORES['danger'], 'fontWeight': 'bold'}),
                        html.Span(f"(valor: {top_negative['valor_atual']:.2f} vs média: {top_negative['media']:.2f})",
                                 style={'color': CORES['text_secondary'], 'fontSize': '12px'})
                    ], style={'marginBottom': '5px'})
                )
            
            resultado_analise.append(html.Hr(style={'borderColor': CORES['border']}))
        
        # ============================================
        # TENDÊNCIA
        # ============================================
        if tendencias:
            resultado_analise.append(
                html.H5("📊 Tendência atual", style={'color': CORES['text']})
            )
            
            for simbolo, feature, status in tendencias[:3]:  # Mostrar top 3
                nome_feature = data_manager.traduzir_coluna(feature)
                cor_tendencia = CORES['success'] if 'acima' in status else (CORES['danger'] if 'abaixo' in status else CORES['text_secondary'])
                resultado_analise.append(
                    html.Div([
                        html.Span(f"{simbolo} ", style={'color': cor_tendencia}),
                        html.Span(f"{nome_feature}: ", style={'color': CORES['text_secondary']}),
                        html.Span(f"{status}", style={'color': cor_tendencia})
                    ], style={'marginBottom': '3px'})
                )
            
            resultado_analise.append(html.Hr(style={'borderColor': CORES['border']}))
        
        # ============================================
        # MENSAGEM E RECOMENDAÇÃO
        # ============================================
        resultado_analise.extend([
            html.Div([
                html.H5("💡 Análise", style={'color': CORES['text']}),
                html.P(
                    nivel['mensagem'],
                    style={'color': CORES['text_secondary']}
                )
            ]),
            
            html.Hr(style={'borderColor': CORES['border']}),
            
            html.Div([
                html.H5("🎯 Recomendação", style={'color': CORES['text']}),
                html.Div([
                    html.P(
                        nivel['recomendacao'],
                        style={'color': cor, 'fontWeight': 'bold', 'fontSize': '16px'}
                    ),
                ], style={'backgroundColor': 'rgba(255,255,255,0.05)', 'padding': '15px', 'borderRadius': '8px'})
            ]),
            
            html.Hr(style={'borderColor': CORES['border']}),
            
            # Valores analisados
            html.H5("📋 Valores analisados", style={'color': CORES['text']}),
            html.Div([
                html.Div([
                    html.Span(nome, style={'color': CORES['text_secondary']}),
                    html.Span(f"{dados[f]:.2f}", style={'color': CORES['text'], 'float': 'right'})
                ], style={'marginBottom': '5px'})
                for nome, f in zip(
                    [data_manager.traduzir_coluna(f) for f in features],
                    features
                )
            ])
        ])
        
        # ============================================
        # LAYOUT FINAL
        # ============================================
        return html.Div([
            dbc.Card([
                dbc.CardBody(resultado_analise)
            ], style={'backgroundColor': CORES['card_bg'], 'border': f'2px solid {cor}'})
        ])
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return html.Div([
            html.H4("Erro na análise", style={'color': CORES['danger']}),
            html.P(f"Erro: {str(e)}", style={'color': CORES['text_secondary']})
        ])