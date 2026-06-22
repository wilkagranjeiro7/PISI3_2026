# ==================================================
# pages/insights.py - ANÁLISE DE INSIGHTS (CORRIGIDA)
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
# FEATURES: separar brutas (input manual) de derivadas (calculadas)
# ==================================================
FEATURES_BRUTAS = ['hrv', 'resting_heart_rate', 'day_strain',
                    'sleep_hours', 'sleep_efficiency', 'sleep_quality']

FEATURES_DERIVADAS = ['strain_per_sleep', 'hrv_rhr_ratio', 'hrv_ratio']


def calcular_features_derivadas(hrv, resting_heart_rate, day_strain, sleep_hours):
    """
    Calcula as features derivadas a partir dos valores brutos.
    IMPORTANTE: essas fórmulas precisam ser IDÊNTICAS às usadas
    na criação do dataset/treinamento do modelo. Confira no seu
    notebook/script de pré-processamento se a fórmula bate.
    """
    hrv_rhr_ratio = hrv / resting_heart_rate if resting_heart_rate > 0 else 0
    strain_per_sleep = day_strain / sleep_hours if sleep_hours > 0 else 0

    # ATENÇÃO: ajuste esta fórmula conforme a definição original do seu dataset.
    # Um exemplo comum é hrv_ratio = hrv atual / hrv médio histórico do usuário.
    # Como não temos um "histórico" aqui, uma aproximação razoável é usar
    # a própria relação hrv/rhr normalizada, ou repetir hrv_rhr_ratio.
    # Troque pela fórmula real usada no treino.
    hrv_ratio = hrv_rhr_ratio

    return strain_per_sleep, hrv_rhr_ratio, hrv_ratio


# ==================================================
# LAYOUT
# ==================================================
def create_layout(df):
    """Layout da página de Insights"""
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

    # Criar inputs SOMENTE para as features brutas (não para as derivadas)
    inputs = []
    for feature in features:
        if feature in FEATURES_DERIVADAS:
            continue  # essas serão calculadas automaticamente, sem input manual

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
def determinar_nivel_recuperacao(probabilidade_overstrain):
    """
    Determina o nível de risco e a recomendação de treino.
    probabilidade_overstrain perto de 1 = alto risco de estafa = NÃO treinar.
    probabilidade_overstrain perto de 0 = corpo recuperado = PODE treinar.
    """
    if probabilidade_overstrain >= 0.7:
        return {
            'nivel': '⚠️ ALTO RISCO DE OVERSTRAIN',
            'cor': '#EF4444',
            'icone': '🚨',
            'pode_treinar': False,
            'mensagem': 'Seu corpo apresenta fortes sinais de sobrecarga fisiológica, estresse cardiovascular acumulado e déficit de descanso.',
            'recomendacao': '🚫 NÃO É RECOMENDADO TREINAR HOJE. Priorize descanso total ou, no máximo, uma atividade regenerativa muito leve (caminhada, mobilidade).',
            'status': 'critico'
        }
    elif probabilidade_overstrain >= 0.35:
        return {
            'nivel': '🟡 ATENÇÃO: FADIGA MODERADA',
            'cor': '#F59E0B',
            'icone': '⚠️',
            'pode_treinar': None,  # treino condicional
            'mensagem': 'Há indícios de desgaste biológico moderado. Seu corpo tolera estímulos, mas a homeostase está limítrofe.',
            'recomendacao': '⚠️ TREINO COM CAUTELA. Prefira intensidade moderada/técnica e evite recordes de carga hoje.',
            'status': 'moderado'
        }
    else:
        return {
            'nivel': '✅ SISTEMA FISIOLÓGICO RECUPERADO',
            'cor': '#10B981',
            'icone': '🟢',
            'pode_treinar': True,
            'mensagem': 'Excelente! Seus indicadores cardiovasculares (VFC) e de sono mostram que o corpo assimilou os treinos anteriores.',
            'recomendacao': '✅ PODE TREINAR. Seu sistema neuromuscular e cardiovascular está pronto para estímulos de alta intensidade!',
            'status': 'excelente'
        }


def analisar_feature_importance(feature_importance, features, dados_atuais, df_referencia):
    if not feature_importance or len(feature_importance) != len(features):
        return None

    df_importancia = pd.DataFrame({
        'feature': features,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    insights = []
    for _, row in df_importancia.iterrows():
        feature = row['feature']
        importance = row['importance']
        valor_atual = dados_atuais.get(feature, 0)
        media = df_referencia[feature].mean() if feature in df_referencia.columns else 0
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
    [State(f'input-{f}', 'value') for f in FEATURES_BRUTAS],
    prevent_initial_call=True
)
def fazer_analise(n_clicks, hrv, resting_heart_rate, day_strain,
                   sleep_hours, sleep_efficiency, sleep_quality):
    """Faz a análise com os dados inseridos usando o modelo salvo"""
    if n_clicks is None:
        return html.Div(
            html.P("Preencha os dados e clique em 'Analisar Recuperação' para obter insights",
                  style={'color': CORES['text_secondary'], 'textAlign': 'center', 'marginTop': '50px'})
        )

    model, scaler, features, metadata = model_manager.carregar_melhor_modelo()
    if model is None:
        return html.Div([
            html.H4("Erro: Nenhum modelo disponível", style={'color': CORES['danger']}),
            html.P("Treine e salve um modelo na página de Classificação primeiro.",
                  style={'color': CORES['text_secondary']})
        ])

    try:
        # Valores brutos vindos do formulário
        hrv = float(hrv) if hrv is not None else 0.0
        resting_heart_rate = float(resting_heart_rate) if resting_heart_rate is not None else 0.0
        day_strain = float(day_strain) if day_strain is not None else 0.0
        sleep_hours = float(sleep_hours) if sleep_hours is not None else 0.0
        sleep_efficiency = float(sleep_efficiency) if sleep_efficiency is not None else 0.0
        sleep_quality = float(sleep_quality) if sleep_quality is not None else 0.0

        # Features derivadas SEMPRE calculadas, nunca digitadas pelo usuário
        strain_per_sleep, hrv_rhr_ratio, hrv_ratio = calcular_features_derivadas(
            hrv, resting_heart_rate, day_strain, sleep_hours
        )

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

        dados = {feature: valores_input.get(feature, 0.0) for feature in features}

        df_usuario = pd.DataFrame([dados])
        X = df_usuario[features].copy()

        if scaler is not None:
            X_scaled = scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=features)

        if hasattr(model, 'predict_proba'):
            proba_overstrain = float(model.predict_proba(X)[0, 1])
        else:
            predicao = int(model.predict(X)[0])
            proba_overstrain = 1.0 if predicao == 1 else 0.0

        score_prontidao = (1 - proba_overstrain) * 100
        nivel = determinar_nivel_recuperacao(proba_overstrain)

        df_referencia = data_manager.get_clean_df()

        feature_importance = metadata.get('metricas', {}).get('feature_importance', [])
        insights_features = None
        if feature_importance and len(feature_importance) == len(features):
            insights_features = analisar_feature_importance(
                feature_importance, features, dados, df_referencia
            )

        tendencias = calcular_tendencia(dados, df_referencia, features)

        cor = nivel['cor']
        icone = nivel['icone']

        # ============================================
        # BANNER DE DECISÃO: pode treinar ou não
        # ============================================
        if nivel['pode_treinar'] is True:
            banner_texto = "✅ PODE TREINAR HOJE"
            banner_cor = CORES['success']
        elif nivel['pode_treinar'] is False:
            banner_texto = "🚫 NÃO É RECOMENDADO TREINAR HOJE"
            banner_cor = CORES['danger']
        else:
            banner_texto = "⚠️ TREINAR COM CAUTELA"
            banner_cor = CORES['warning']

        resultado_analise = [
            html.Div([
                html.H2(
                    banner_texto,
                    style={'color': banner_cor, 'fontWeight': 'bold', 'textAlign': 'center',
                          'padding': '15px', 'border': f'2px solid {banner_cor}', 'borderRadius': '10px',
                          'marginBottom': '20px'}
                ),
            ]),
            html.Div([
                html.H4(f"{icone} Avaliação Fisiológica", style={'color': CORES['text']}),
                html.H3(
                    nivel['nivel'],
                    style={'color': cor, 'marginTop': '10px', 'fontWeight': 'bold'}
                ),
            ]),
            html.Div([
                html.P(
                    f"Índice de Prontidão para o Treino: {score_prontidao:.1f} / 100",
                    style={'color': CORES['text_secondary'], 'fontSize': '18px', 'marginTop': '10px'}
                ),
                html.Div([
                    html.Div(
                        style={
                            'width': f"{max(score_prontidao, 5)}%",
                            'height': '30px',
                            'backgroundColor': cor,
                            'borderRadius': '15px',
                            'transition': 'width 0.5s',
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
                f"Probabilidade Estatística de Overstrain: {proba_overstrain:.1%}",
                style={'color': CORES['text_secondary'], 'fontSize': '14px', 'marginTop': '10px'}
            ),
            html.Hr(style={'borderColor': CORES['border']}),
        ]

        if insights_features:
            positivas = [i for i in insights_features if i['impacto'] == 'positivo']
            negativas = [i for i in insights_features if i['impacto'] == 'negativo']

            resultado_analise.append(html.H5("🔍 Fatores que mais influenciaram", style={'color': CORES['text']}))

            if positivas:
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

        if tendencias:
            resultado_analise.append(html.H5("📊 Tendência atual", style={'color': CORES['text']}))
            for simbolo, feature, status in tendencias[:3]:
                nome_feature = data_manager.traduzir_coluna(feature)
                cor_tendencia = CORES['success'] if 'acima' in status else (
                    CORES['danger'] if 'abaixo' in status else CORES['text_secondary'])
                resultado_analise.append(
                    html.Div([
                        html.Span(f"{simbolo} ", style={'color': cor_tendencia}),
                        html.Span(f"{nome_feature}: ", style={'color': CORES['text_secondary']}),
                        html.Span(f"{status}", style={'color': cor_tendencia})
                    ], style={'marginBottom': '3px'})
                )
            resultado_analise.append(html.Hr(style={'borderColor': CORES['border']}))

        resultado_analise.extend([
            html.Div([
                html.H5("💡 Análise", style={'color': CORES['text']}),
                html.P(nivel['mensagem'], style={'color': CORES['text_secondary']})
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