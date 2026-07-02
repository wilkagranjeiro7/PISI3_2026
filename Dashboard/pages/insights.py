# ==================================================
# pages/insights.py - ANÁLISE DE INSIGHTS (CORRIGIDO)
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
    
    # Criar inputs para cada feature (CAMPOS TOTALMENTE LIMPOS)
    inputs = []
    for feature in features:
        nome_traduzido = data_manager.traduzir_coluna(feature)
        
        inputs.append(html.Div([
            html.Label(nome_traduzido, style={'color': CORES['text'], 'fontSize': '14px', 'marginBottom': '5px'}),
            dbc.Input(
                type="number",
                id=f"input-{feature}",
                value=None,  # Campo vazio - SEM valor pré-preenchido
                step=0.01,
                style={
                    'backgroundColor': CORES['card_bg'], 
                    'color': CORES['text'], 
                    'border': f'1px solid {CORES["border"]}',
                    'width': '100%'
                }
            )
        ], style={'marginBottom': '15px'}))
    
    return html.Div([
        html.Div([
            dbc.Button("← Voltar", href="/", color="light", size="sm",
                      style={'backgroundColor': 'transparent', 'border': f'1px solid {CORES["border"]}', 
                            'color': CORES['text']})
        ], style={'position': 'fixed', 'top': '20px', 'left': '20px', 'zIndex': '1000'}),
        
        html.Div([
            html.H3("Análise de Insights para Recuperação", style={'color': CORES['text'], 'marginBottom': '10px'}),
            html.P("Preencha seus dados e entenda como os fatores influenciam sua recuperação",
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
                    html.H4("Informe seus Dados", style={'color': CORES['text'], 'marginBottom': '20px'}),
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


def obter_referencia_hibrida(feature, dados_atuais, metadata, df_referencia=None):
    """
    Abordagem Híbrida para referências usando dados do modelo:
    - Features fisiológicas (HRV, RHR): Percentis do dataset (via metadata)
    - Features de sono: Referência da literatura
    - Features derivadas: Grupo de alta recuperação (via metadata)
    """
    
    # 1. FEATURES DE SONO → LITERATURA CIENTÍFICA
    FEATURES_SONO = ['sleep_hours', 'sleep_efficiency', 'sleep_quality']
    if feature in FEATURES_SONO:
        referencias = {
            'sleep_hours': {'ideal': 8.0, 'min': 7.0, 'max': 9.0},
            'sleep_efficiency': {'ideal': 0.90, 'min': 0.85, 'max': 1.0},
            'sleep_quality': {'ideal': 8.0, 'min': 7.0, 'max': 10.0}
        }
        ref = referencias.get(feature, {})
        return {
            'tipo': 'literatura',
            'referencia': ref.get('ideal', 0),
            'min': ref.get('min', 0),
            'max': ref.get('max', 0),
            'label': 'Recomendação científica'
        }
    
    # 2. FEATURES DERIVADAS → GRUPO DE ALTA RECUPERAÇÃO (do metadata)
    FEATURES_DERIVADAS = ['strain_per_sleep', 'hrv_rhr_ratio', 'hrv_ratio']
    if feature in FEATURES_DERIVADAS:
        # Tentar obter do metadata
        referencias = metadata.get('referencias', {})
        alta_recuperacao = referencias.get('alta_recuperacao', {})
        
        if feature in alta_recuperacao:
            return {
                'tipo': 'alta_recuperacao',
                'referencia': alta_recuperacao[feature],
                'label': 'Média de atletas com alta recuperação'
            }
        
        # Fallback: tentar percentis do metadata
        percentis = referencias.get('percentis', {})
        if feature in percentis:
            return {
                'tipo': 'percentil',
                'referencia': percentis[feature].get('p50', 0),
                'percentil_75': percentis[feature].get('p75', 0),
                'percentil_25': percentis[feature].get('p25', 0),
                'label': 'Percentil 50 do dataset'
            }
        
        # Fallback final: usar df_referencia se disponível
        if df_referencia is not None and feature in df_referencia.columns:
            return {
                'tipo': 'media',
                'referencia': df_referencia[feature].mean(),
                'label': 'Média do dataset'
            }
    
    # 3. FEATURES FISIOLÓGICAS → PERCENTIS DO METADATA
    # (hrv, resting_heart_rate, day_strain)
    referencias = metadata.get('referencias', {})
    percentis = referencias.get('percentis', {})
    
    if feature in percentis:
        p25 = percentis[feature].get('p25', 0)
        p50 = percentis[feature].get('p50', 0)
        p75 = percentis[feature].get('p75', 0)
        
        valor = dados_atuais.get(feature, 0)
        
        # Determinar status baseado no percentil
        if valor >= p75:
            status = 'excelente'
            label = 'Acima de 75% dos atletas'
        elif valor >= p50:
            status = 'bom'
            label = 'Acima da média'
        elif valor >= p25:
            status = 'regular'
            label = 'Na média'
        else:
            status = 'baixo'
            label = 'Abaixo de 25% dos atletas'
        
        return {
            'tipo': 'percentil',
            'referencia': p50,
            'percentil_75': p75,
            'percentil_25': p25,
            'status': status,
            'label': label
        }
    
    # Fallback: usar df_referencia se disponível
    if df_referencia is not None and feature in df_referencia.columns:
        return {
            'tipo': 'media',
            'referencia': df_referencia[feature].mean(),
            'label': 'Média do dataset'
        }
    
    # Fallback final
    return {
        'tipo': 'padrao',
        'referencia': 0,
        'label': 'Valor padrão'
    }


def analisar_feature_importance(feature_importance, features, dados_atuais, metadata, df_referencia=None):
    """
    Analisa quais features mais contribuíram para o resultado
    Usando a abordagem híbrida para referências
    """
    if not feature_importance or len(feature_importance) != len(features):
        return None
    
    # Criar DataFrame com importâncias
    df_importancia = pd.DataFrame({
        'feature': features,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    # Para cada feature, calcular contribuição com referência híbrida
    insights = []
    for _, row in df_importancia.iterrows():
        feature = row['feature']
        importance = row['importance']
        valor_atual = dados_atuais.get(feature, 0)
        
        # Obter referência híbrida (usando metadata)
        ref = obter_referencia_hibrida(feature, dados_atuais, metadata, df_referencia)
        referencia = ref.get('referencia', 0)
        
        # Contribuição baseada na importância e diferença da referência
        contribuicao = importance * (valor_atual - referencia)
        
        insights.append({
            'feature': feature,
            'importance': importance,
            'valor_atual': valor_atual,
            'referencia': referencia,
            'tipo_referencia': ref.get('tipo', 'media'),
            'contribuicao': contribuicao,
            'impacto': 'positivo' if contribuicao > 0 else 'negativo'
        })
    
    return insights


def calcular_tendencia(dados_atuais, metadata, features, df_referencia=None):
    """
    Calcula a tendência baseada nos valores atuais em comparação com a referência híbrida
    """
    tendencias = []
    for feature in features:
        if feature in dados_atuais:
            valor_atual = dados_atuais[feature]
            
            # Obter referência híbrida (usando metadata)
            ref = obter_referencia_hibrida(feature, dados_atuais, metadata, df_referencia)
            referencia = ref.get('referencia', 0)
            
            # Calcular diferença da referência
            diff = valor_atual - referencia
            
            # Para features onde valores menores são melhores (ex: RHR)
            FEATURES_MENOR_MELHOR = ['resting_heart_rate', 'day_strain', 'strain_per_sleep']
            if feature in FEATURES_MENOR_MELHOR:
                diff = -diff  # Inverter a lógica
            
            if diff > 0.5:
                tendencias.append(('📈', feature, 'acima da referência', 'positivo'))
            elif diff < -0.5:
                tendencias.append(('📉', feature, 'abaixo da referência', 'negativo'))
            else:
                tendencias.append(('➡️', feature, 'na referência', 'neutro'))
    
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
            html.P("Preencha seus dados e clique em 'Analisar Recuperação' para obter insights personalizados",
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
    
    # VALIDAÇÃO: Verificar se todos os campos foram preenchidos
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
    
    # Verificar campos vazios
    campos_vazios = [k for k, v in valores_input.items() if v is None or v == '']
    if campos_vazios:
        nomes_campos = [data_manager.traduzir_coluna(f) for f in campos_vazios]
        return html.Div([
            html.H4("⚠️ Campos incompletos", style={'color': CORES['warning']}),
            html.P(f"Preencha os seguintes campos: {', '.join(nomes_campos)}",
                  style={'color': CORES['text_secondary']}),
            html.P("Todos os dados são necessários para uma análise precisa.",
                  style={'color': CORES['text_secondary'], 'fontSize': '12px', 'marginTop': '10px'})
        ], style={'padding': '20px'})
    
    try:
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
        
        # Determinar o nível de recuperação
        nivel = determinar_nivel_recuperacao(proba_alta, score_recuperacao)
        
        # Carregar DataFrame de referência (apenas como fallback)
        df_referencia = data_manager.get_clean_df() if data_manager else None
        
        # ============================================
        # ANÁLISE DE FEATURE IMPORTANCE
        # ============================================
        feature_importance = metadata.get('metricas', {}).get('feature_importance', [])
        insights_features = None
        
        if feature_importance and len(feature_importance) == len(features):
            insights_features = analisar_feature_importance(
                feature_importance, features, dados, metadata, df_referencia
            )
        
        # ============================================
        # TENDÊNCIA COM REFERÊNCIA HÍBRIDA
        # ============================================
        tendencias = calcular_tendencia(dados, metadata, features, df_referencia)
        
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
                html.P("Comparação com referências personalizadas para cada métrica",
                      style={'color': CORES['text_secondary'], 'fontSize': '12px', 'marginBottom': '10px'})
            ])
            
            if positivas:
                # Pegar a feature mais positiva
                top_positive = max(positivas, key=lambda x: abs(x['contribuicao']))
                nome_feature = data_manager.traduzir_coluna(top_positive['feature'])
                ref = obter_referencia_hibrida(top_positive['feature'], dados, metadata, df_referencia)
                tipo_ref = ref.get('label', 'referência')
                
                resultado_analise.append(
                    html.Div([
                        html.Span("🟢 ", style={'color': CORES['success']}),
                        html.Span(f"Contribuição positiva: ", style={'color': CORES['text_secondary']}),
                        html.Span(f"{nome_feature} ", style={'color': CORES['success'], 'fontWeight': 'bold'}),
                        html.Span(f"(valor: {top_positive['valor_atual']:.2f} vs {tipo_ref}: {top_positive['referencia']:.2f})",
                                 style={'color': CORES['text_secondary'], 'fontSize': '12px'})
                    ], style={'marginBottom': '5px'})
                )
            
            if negativas:
                # Pegar a feature mais negativa
                top_negative = max(negativas, key=lambda x: abs(x['contribuicao']))
                nome_feature = data_manager.traduzir_coluna(top_negative['feature'])
                ref = obter_referencia_hibrida(top_negative['feature'], dados, metadata, df_referencia)
                tipo_ref = ref.get('label', 'referência')
                
                resultado_analise.append(
                    html.Div([
                        html.Span("🔴 ", style={'color': CORES['danger']}),
                        html.Span(f"Contribuição negativa: ", style={'color': CORES['text_secondary']}),
                        html.Span(f"{nome_feature} ", style={'color': CORES['danger'], 'fontWeight': 'bold'}),
                        html.Span(f"(valor: {top_negative['valor_atual']:.2f} vs {tipo_ref}: {top_negative['referencia']:.2f})",
                                 style={'color': CORES['text_secondary'], 'fontSize': '12px'})
                    ], style={'marginBottom': '5px'})
                )
            
            resultado_analise.append(html.Hr(style={'borderColor': CORES['border']}))
        
        # ============================================
        # TENDÊNCIA COM REFERÊNCIA HÍBRIDA
        # ============================================
        if tendencias:
            resultado_analise.append(
                html.H5("📊 Tendência atual", style={'color': CORES['text']})
            )
            
            for simbolo, feature, status, tipo in tendencias[:3]:  # Mostrar top 3
                nome_feature = data_manager.traduzir_coluna(feature)
                if tipo == 'positivo':
                    cor_tendencia = CORES['success']
                elif tipo == 'negativo':
                    cor_tendencia = CORES['danger']
                else:
                    cor_tendencia = CORES['text_secondary']
                    
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
            
            # Valores analisados com referências
            html.H5("📋 Valores analisados", style={'color': CORES['text']}),
        ])
        
        # Adicionar cada valor com sua referência
        for feature in features:
            nome_feature = data_manager.traduzir_coluna(feature)
            valor = dados.get(feature, 0)
            
            # Obter referência híbrida para mostrar
            ref = obter_referencia_hibrida(feature, dados, metadata, df_referencia)
            referencia = ref.get('referencia', 0)
            tipo_ref = ref.get('label', 'referência')
            
            # Cor baseada na comparação com referência
            diff = valor - referencia
            FEATURES_MENOR_MELHOR = ['resting_heart_rate', 'day_strain', 'strain_per_sleep']
            if feature in FEATURES_MENOR_MELHOR:
                diff = -diff
            
            if diff > 0.5:
                cor_valor = CORES['success']
            elif diff < -0.5:
                cor_valor = CORES['danger']
            else:
                cor_valor = CORES['text_secondary']
            
            resultado_analise.append(
                html.Div([
                    html.Div([
                        html.Span(nome_feature, style={'color': CORES['text_secondary']}),
                        html.Span(f"{valor:.2f}", style={'color': cor_valor, 'float': 'right', 'fontWeight': 'bold'})
                    ], style={'marginBottom': '2px'}),
                    html.Div(
                        f"↳ {tipo_ref}: {referencia:.2f}",
                        style={'color': CORES['text_secondary'], 'fontSize': '11px', 'marginBottom': '8px', 'paddingLeft': '10px'}
                    )
                ])
            )
        
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
            html.H4("❌ Erro na análise", style={'color': CORES['danger']}),
            html.P(f"Ocorreu um erro ao processar seus dados: {str(e)}", 
                  style={'color': CORES['text_secondary']}),
            html.P("Verifique se todos os valores são números válidos.",
                  style={'color': CORES['text_secondary'], 'fontSize': '12px', 'marginTop': '10px'})
        ])