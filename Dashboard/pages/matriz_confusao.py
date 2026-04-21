from dash import html, dcc, Input, Output
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, matthews_corrcoef
import numpy as np

def create_layout(df=None):
    """Página da matriz de confusão"""
    
    return html.Div([
        html.H1("Matriz de Confusão", style={'marginBottom': 20, 'color': '#2c3e50'}),
        
        # Card do botão
        html.Div([
            html.Button("📊 Gerar Matriz de Confusão", 
                       id="botao-gerar-matriz", 
                       n_clicks=0,
                       style={
                           'backgroundColor': '#3498db',
                           'color': 'white',
                           'border': 'none',
                           'padding': '12px 24px',
                           'borderRadius': '6px',
                           'cursor': 'pointer',
                           'fontSize': '16px',
                           'fontWeight': 'bold'
                       }),
        ], style={'textAlign': 'center', 'marginBottom': 30}),
        
        # Container para os resultados
        html.Div(id='matriz-grafico-output', className="chart-card"),
        
        # Substituir html.Style por um dcc.Markdown ou remover
        # Os estilos serão aplicados via inline styles ou arquivo CSS externo
    ])

def register_callbacks(app):
    """Registra os callbacks da página"""
    
    @app.callback(
        Output('matriz-grafico-output', 'children'),
        Input('botao-gerar-matriz', 'n_clicks'),
        prevent_initial_call=False
    )
    def update_matriz_confusao(n_clicks):
        if n_clicks == 0:
            return html.Div(
                html.Div([
                    html.Div("🔘", style={'fontSize': 48, 'marginBottom': 20}),
                    html.Div("Clique no botão acima para gerar a matriz de confusão", 
                            style={'fontSize': 18, 'color': '#7f8c8d'}),
                    html.Div("O modelo será treinado com os dados de recovery score", 
                            style={'fontSize': 14, 'color': '#95a5a6', 'marginTop': 10})
                ], style={'textAlign': 'center', 'padding': 60}),
                style={
                    'background': 'white',
                    'borderRadius': '10px',
                    'padding': '20px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'marginBottom': '20px'
                }
            )
        
        try:
            # Carregar dados
            from data_loader import data_manager
            df = data_manager.df.dropna()
            
            # Preparar dados
            df['target'] = (df['recovery_score'] > 66).astype(int)
            features = ['day_strain', 'sleep_hours', 'sleep_efficiency', 'hrv', 'resting_heart_rate']
            X = df[features]
            y = df['target']
            
            # Verificar se há dados suficientes
            if len(X) < 10:
                return html.Div([
                    html.H3("⚠️ Dados Insuficientes", style={'color': '#e74c3c'}),
                    html.P(f"Apenas {len(X)} amostras disponíveis. São necessárias pelo menos 10 amostras.")
                ], style={
                    'background': '#fff3cd',
                    'border': '1px solid #ffc107',
                    'color': '#856404',
                    'padding': '15px',
                    'borderRadius': '5px',
                    'margin': '10px 0'
                })
            
            # Verificar balanceamento das classes
            class_balance = y.value_counts()
            total_samples = len(y)
            class0_percent = (class_balance[0] / total_samples) * 100 if 0 in class_balance else 0
            class1_percent = (class_balance[1] / total_samples) * 100 if 1 in class_balance else 0
            
            # Treinar modelo
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            clf.fit(X_train, y_train)
            
            # Previsões
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]
            cm = confusion_matrix(y_test, y_pred)
            
            # Calcular todas as métricas
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            mcc = matthews_corrcoef(y_test, y_pred)
            
            # AUC-ROC
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            # Análise de erros
            erro_tipo1 = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0
            erro_tipo2 = (fn / (fn + tp) * 100) if (fn + tp) > 0 else 0
            
            # Acurácia por classe
            acuracia_classe0 = tn / (tn + fp) if (tn + fp) > 0 else 0
            acuracia_classe1 = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # Melhor threshold
            youden_j = tpr - fpr
            best_threshold_idx = np.argmax(youden_j)
            best_threshold = thresholds[best_threshold_idx] if len(thresholds) > 0 else 0.5
            
            # Calcular NPV
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            # ==================== VISUALIZAÇÕES ====================
            
            # 1. Matriz de Confusão Heatmap
            fig_matriz = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Baixo Recovery', 'Alto Recovery'],
                y=['Baixo Recovery', 'Alto Recovery'],
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 20, "color": "white"},
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Quantidade")
            ))
            
            fig_matriz.update_layout(
                title="<b>Matriz de Confusão - Classificação de Recovery Score</b>",
                xaxis_title="<b>Predito</b>",
                yaxis_title="<b>Real</b>",
                template='plotly_white',
                height=550,
                width=650
            )
            
            # 2. Gráfico de Barras das Métricas
            metricas_dict = {
                'Acurácia': accuracy, 
                'Precisão': precision, 
                'Recall': recall,
                'Especificidade': specificity,
                'F1-Score': f1
            }
            
            colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
            fig_metricas = go.Figure(data=[
                go.Bar(
                    x=list(metricas_dict.keys()),
                    y=list(metricas_dict.values()),
                    text=[f'{v:.1%}' for v in metricas_dict.values()],
                    textposition='outside',
                    marker_color=colors,
                    opacity=0.8
                )
            ])
            
            fig_metricas.update_layout(
                title="<b>Métricas de Desempenho do Modelo</b>",
                xaxis_title="<b>Métrica</b>",
                yaxis_title="<b>Valor</b>",
                template='plotly_white',
                height=450,
                yaxis_tickformat='.0%',
                yaxis_range=[0, 1.1]
            )
            
            # 3. Curva ROC
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'Modelo (AUC = {roc_auc:.3f})',
                line=dict(color='#3498db', width=3),
                fill='tozeroy',
                fillcolor='rgba(52, 152, 219, 0.2)'
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Classificador Aleatório',
                line=dict(color='red', width=2, dash='dash')
            ))
            fig_roc.update_layout(
                title="<b>Curva ROC - Performance do Modelo</b>",
                xaxis_title="<b>Taxa de Falsos Positivos (1 - Especificidade)</b>",
                yaxis_title="<b>Taxa de Verdadeiros Positivos (Recall)</b>",
                template='plotly_white',
                height=450,
                width=550
            )
            
            # 4. Importância das Features
            feature_importance = clf.feature_importances_
            features_names = ['Estresse Diário', 'Horas Sono', 'Eficiência Sono', 'HRV', 'FC Repouso']
            
            fig_importancia = go.Figure(data=[
                go.Bar(
                    x=feature_importance,
                    y=features_names,
                    orientation='h',
                    text=[f'{imp:.2%}' for imp in feature_importance],
                    textposition='outside',
                    marker_color='#3498db'
                )
            ])
            
            fig_importancia.update_layout(
                title="<b>Importância das Features para o Modelo</b>",
                xaxis_title="<b>Importância</b>",
                yaxis_title="<b>Feature</b>",
                template='plotly_white',
                height=400,
                xaxis_tickformat='.0%'
            )
            
            # Recomendações
            recomendacoes = []
            if erro_tipo1 > 20:
                recomendacoes.append("🔧 Alto número de Falsos Positivos")
            if erro_tipo2 > 20:
                recomendacoes.append("📊 Alto número de Falsos Negativos")
            if precision < 0.7:
                recomendacoes.append("🎯 Precisão baixa")
            if recall < 0.7:
                recomendacoes.append("🔄 Recall baixo")
            
            # Determinar nível de performance
            if accuracy > 0.85:
                performance_level = "Excelente"
                performance_color = "#28a745"
                performance_icon = "✅"
            elif accuracy > 0.7:
                performance_level = "Bom"
                performance_color = "#3498db"
                performance_icon = "👍"
            else:
                performance_level = "Regular"
                performance_color = "#f39c12"
                performance_icon = "⚠️"
            
            # ==================== LAYOUT ====================
            
            return html.Div([
                # Resumo
                html.Div([
                    html.H2(f"{performance_icon} Performance: {performance_level}", 
                            style={'color': performance_color}),
                    html.P(f"Acurácia: {accuracy:.1%}")
                ], style={
                    'background': '#f8f9fa',
                    'borderLeft': f'4px solid {performance_color}',
                    'padding': '15px',
                    'borderRadius': '5px',
                    'margin': '10px 0'
                }),
                
                # Matriz e Métricas
                html.Div([
                    html.Div([
                        html.H3("Matriz de Confusão", style={'textAlign': 'center'}),
                        dcc.Graph(figure=fig_matriz, config={'displayModeBar': False})
                    ], style={'flex': 1, 'marginRight': '15px'}),
                    
                    html.Div([
                        html.H3("Métricas", style={'textAlign': 'center'}),
                        html.Div([
                            html.Div([
                                html.Div(f"{accuracy:.1%}", style={'fontSize': '32px', 'fontWeight': 'bold'}),
                                html.Div("Acurácia")
                            ], style={
                                'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                                'color': 'white',
                                'padding': '20px',
                                'borderRadius': '10px',
                                'textAlign': 'center'
                            }),
                            html.Div([
                                html.Div(f"{precision:.1%}", style={'fontSize': '32px', 'fontWeight': 'bold'}),
                                html.Div("Precisão")
                            ], style={
                                'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                                'color': 'white',
                                'padding': '20px',
                                'borderRadius': '10px',
                                'textAlign': 'center'
                            })
                        ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(2, 1fr)', 'gap': '20px'})
                    ], style={'flex': 1})
                ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '30px'}),
                
                # Detalhamento
                html.Div([
                    html.H3("Detalhamento"),
                    html.Div([
                        html.Div(f"✅ VP: {tp}", style={'color': '#28a745'}),
                        html.Div(f"❌ FP: {fp}", style={'color': '#dc3545'}),
                        html.Div(f"❌ FN: {fn}", style={'color': '#dc3545'}),
                        html.Div(f"✅ VN: {tn}", style={'color': '#28a745'})
                    ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 'gap': '15px'})
                ], style={
                    'background': 'white',
                    'borderRadius': '10px',
                    'padding': '20px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'marginBottom': '20px'
                }),
                
                # Gráficos
                html.Div([
                    dcc.Graph(figure=fig_metricas, config={'displayModeBar': False})
                ], style={
                    'background': 'white',
                    'borderRadius': '10px',
                    'padding': '20px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'marginBottom': '20px'
                }),
                
                html.Div([
                    dcc.Graph(figure=fig_roc, config={'displayModeBar': False})
                ], style={
                    'background': 'white',
                    'borderRadius': '10px',
                    'padding': '20px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'marginBottom': '20px'
                }),
                
                html.Div([
                    dcc.Graph(figure=fig_importancia, config={'displayModeBar': False})
                ], style={
                    'background': 'white',
                    'borderRadius': '10px',
                    'padding': '20px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'marginBottom': '20px'
                }),
                
                # Recomendações
                html.Div([
                    html.H3("Recomendações"),
                    html.Ul([html.Li(rec) for rec in recomendacoes]) if recomendacoes else html.P("✅ Modelo OK")
                ], style={
                    'background': '#e8f4fd',
                    'borderRadius': '10px',
                    'padding': '20px',
                    'marginBottom': '20px'
                })
                
            ])
            
        except Exception as e:
            return html.Div([
                html.H3("❌ Erro", style={'color': '#e74c3c'}),
                html.P(f"Erro: {str(e)}")
            ], style={
                'background': '#fff3cd',
                'border': '1px solid #ffc107',
                'color': '#856404',
                'padding': '15px',
                'borderRadius': '5px'
            })