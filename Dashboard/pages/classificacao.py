# pages/classificacao.py
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, callback, State
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from data_loader import data_manager


# ==================================================
# CORES PADRÃO (via DataManager)
# ==================================================

CORES = data_manager.get_cores()

# Cores para as métricas (baseado na paleta)
METRIC_COLORS = [CORES['accent'], CORES['success'], CORES['warning'], CORES['hrv'], CORES['danger'], CORES['text_secondary']]

CLASSIFIERS = {
    'Random Forest': {
        'model': RandomForestClassifier,
        'default': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier,
        'default': {'n_estimators': 100, 'random_state': 42}
    },
    'Logistic Regression': {
        'model': LogisticRegression,
        'default': {'random_state': 42, 'max_iter': 1000}
    },
    'SVM': {
        'model': SVC,
        'default': {'random_state': 42, 'probability': True}
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier,
        'default': {'random_state': 42}
    },
    'KNN': {
        'model': KNeighborsClassifier,
        'default': {}
    }
}


def create_layout(df):
    """Layout da página de classificação"""
    
    features = ['day_strain', 'sleep_hours', 'sleep_efficiency', 'hrv', 'resting_heart_rate']
    
    features_options = [
        {'label': data_manager.traduzir_coluna('day_strain'), 'value': 'day_strain'},
        {'label': data_manager.traduzir_coluna('sleep_hours'), 'value': 'sleep_hours'},
        {'label': data_manager.traduzir_coluna('sleep_efficiency'), 'value': 'sleep_efficiency'},
        {'label': data_manager.traduzir_coluna('hrv'), 'value': 'hrv'},
        {'label': data_manager.traduzir_coluna('resting_heart_rate'), 'value': 'resting_heart_rate'},
    ]
    
    models_options = [
        {'label': 'Random Forest', 'value': 'Random Forest'},
        {'label': 'Gradient Boosting', 'value': 'Gradient Boosting'},
        {'label': 'Regressão Logística', 'value': 'Logistic Regression'},
        {'label': 'SVM', 'value': 'SVM'},
        {'label': 'Árvore de Decisão', 'value': 'Decision Tree'},
        {'label': 'KNN', 'value': 'KNN'},
    ]
    
    return html.Div([
        # Botão voltar
        html.Div([
            dbc.Button("← Voltar", href="/", color="light", size="sm",
                      style={'backgroundColor': 'transparent', 'border': f'1px solid {CORES["border"]}', 
                            'color': CORES['text']})
        ], style={'position': 'fixed', 'top': '20px', 'left': '20px', 'zIndex': '1000'}),
        
        # Conteúdo principal
        html.Div([
            # Painel esquerdo - configurações
            html.Div([
                html.H3("Classificação", style={'fontWeight': 'normal', 'marginBottom': '30px', 'color': CORES['text']}),
                
                html.Div([
                    html.Label("CARACTERÍSTICAS", style={'color': CORES['text_secondary'], 'fontSize': '12px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                    html.Div([
                        dbc.Checklist(
                            id='class-features',
                            options=features_options,
                            value=['day_strain', 'sleep_hours', 'sleep_efficiency', 'hrv', 'resting_heart_rate'],
                            inline=False,
                            switch=True,
                            style={'marginTop': '10px'}
                        )
                    ])
                ], style={'marginBottom': '30px'}),
                
                html.Div([
                    html.Label("TAMANHO DO TESTE", style={'color': CORES['text_secondary'], 'fontSize': '12px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                    html.Div([
                        dbc.RadioItems(
                            id='class-test-size',
                            options=[
                                {'label': '10%', 'value': 0.1},
                                {'label': '20%', 'value': 0.2},
                                {'label': '30%', 'value': 0.3},
                            ],
                            value=0.2,
                            inline=True,
                            style={'marginTop': '10px'}
                        )
                    ])
                ], style={'marginBottom': '30px'}),
                
                html.Div([
                    html.Label("MODELOS", style={'color': CORES['text_secondary'], 'fontSize': '12px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                    html.Div([
                        dbc.Checklist(
                            id='class-models',
                            options=models_options,
                            value=['Random Forest', 'Gradient Boosting'],
                            style={'marginTop': '10px'}
                        )
                    ])
                ], style={'marginBottom': '30px'}),
                
                html.Div([
                    html.Label("HIPERPARÂMETROS", style={'color': CORES['text_secondary'], 'fontSize': '12px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                    html.Div([
                        dbc.Checkbox(
                            id='class-optimize',
                            label="Otimizar com Grid Search",
                            value=False,
                            style={'marginTop': '10px'}
                        )
                    ])
                ], style={'marginBottom': '30px'}),
                
                dbc.Button(
                    "Executar", 
                    id='class-run-button', 
                    color="primary", 
                    size="sm",
                    className="w-100 mt-3",
                    style={'backgroundColor': CORES['accent'], 'border': 'none'}
                )
                
            ], style={
                'position': 'fixed', 
                'width': '300px', 
                'padding': '80px 25px 20px 25px',
                'borderRight': f'1px solid {CORES["border"]}',
                'height': '100vh',
                'overflowY': 'auto',
                'backgroundColor': CORES['background']
            }),
            
            # Painel direito - resultados
            html.Div([
                dcc.Store(id='class-results-store'),
                html.Div(id='class-results-container', children=[
                    html.Div([
                        html.P("Selecione os modelos e clique em Executar", 
                              style={'color': CORES['text_secondary'], 'textAlign': 'center', 'marginTop': '50px'})
                    ])
                ])
            ], style={'marginLeft': '320px', 'padding': '20px', 'minHeight': '100vh'})
            
        ])
        
    ], style={'backgroundColor': CORES['background'], 'minHeight': '100vh', 'color': CORES['text']})


@callback(
    [Output('class-results-container', 'children'),
     Output('class-results-store', 'data')],
    Input('class-run-button', 'n_clicks'),
    State('class-features', 'value'),
    State('class-test-size', 'value'),
    State('class-models', 'value'),
    State('class-optimize', 'value'),
    prevent_initial_call=True
)
def run_classification(n_clicks, features, test_size, selected_models, optimize):
    if not selected_models:
        return html.P("⚠️ Selecione pelo menos um modelo", style={'color': CORES['warning']}), {}
    
    if len(features) < 2:
        return html.P("⚠️ Selecione pelo menos 2 características", style={'color': CORES['warning']}), {}
    
    try:
        df = data_manager.get_clean_df()
        
        if df is None:
            return html.P("❌ Dados não disponíveis.", style={'color': CORES['danger']}), {}
        
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            return html.P(f"❌ Features não encontradas: {missing_features}", style={'color': CORES['danger']}), {}
        
        if 'recovery_score' not in df.columns:
            return html.P("❌ Coluna 'recovery_score' não encontrada", style={'color': CORES['danger']}), {}
        
        # Selecionar colunas e remover nulos
        df_clean = df[features + ['recovery_score']].dropna().copy()
        
        if df_clean.empty:
            return html.P("❌ Dados insuficientes", style={'color': CORES['danger']}), {}
        
        # Garantir que recovery_score é numérico
        df_clean['recovery_score'] = pd.to_numeric(df_clean['recovery_score'], errors='coerce')
        df_clean = df_clean.dropna(subset=['recovery_score'])
        
        # Criar target
        df_clean['target'] = (df_clean['recovery_score'] > 66).astype(int)
        
        X = df_clean[features]
        y = df_clean['target']
        
        if len(y.unique()) < 2:
            return html.P("⚠️ Dados possuem apenas uma classe", style={'color': CORES['warning']}), {}
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        results = []
        
        for model_name in selected_models:
            model_config = CLASSIFIERS[model_name]
            model = model_config['model'](**model_config['default'])
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            results.append({
                'model': model_name,
                'accuracy': accuracy,
                'f1_score': report.get('weighted avg', {}).get('f1-score', 0),
                'precision': report.get('weighted avg', {}).get('precision', 0),
                'recall': report.get('weighted avg', {}).get('recall', 0),
                'feature_importances': model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else None,
            })
        
        results.sort(key=lambda x: x['f1_score'], reverse=True)
        best_model = results[0]
        
        store_data = {
            'results': results,
            'features': features
        }
        
        # Gráfico de acurácia
        df_plot = pd.DataFrame(results)
        fig_comparison = px.bar(df_plot, x='model', y='accuracy', 
                                 title="Acurácia por Modelo",
                                 text=[f'{v:.1%}' for v in df_plot['accuracy']],
                                 color='accuracy',
                                 color_continuous_scale=['#888888', CORES['accent']])
        fig_comparison.update_traces(textposition='outside')
        fig_comparison.update_layout(
            template='plotly_dark',
            paper_bgcolor=CORES['card_bg'],
            plot_bgcolor=CORES['card_bg'],
            font_color=CORES['text'],
            showlegend=False,
            height=450,
            yaxis=dict(title="Acurácia", range=[0, 1], gridcolor=CORES['border']),
            xaxis=dict(title="Modelo", gridcolor=CORES['border'])
        )
        
        # Gráfico de métricas
        metrics_data = []
        for res in results:
            metrics_data.append({'Modelo': res['model'], 'Métrica': 'Precisão', 'Valor': res['precision']})
            metrics_data.append({'Modelo': res['model'], 'Métrica': 'Recall', 'Valor': res['recall']})
            metrics_data.append({'Modelo': res['model'], 'Métrica': 'F1-Score', 'Valor': res['f1_score']})
        
        df_metrics = pd.DataFrame(metrics_data)
        
        fig_metrics = px.bar(df_metrics, x='Modelo', y='Valor', color='Métrica',
                              barmode='group',
                              title="Métricas por Modelo",
                              color_discrete_map={
                                  'Precisão': METRIC_COLORS[0],
                                  'Recall': METRIC_COLORS[1],
                                  'F1-Score': METRIC_COLORS[2]
                              },
                              text='Valor')
        fig_metrics.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_metrics.update_layout(
            template='plotly_dark',
            paper_bgcolor=CORES['card_bg'],
            plot_bgcolor=CORES['card_bg'],
            font_color=CORES['text'],
            height=500,
            yaxis=dict(title="Valor", range=[0, 1], gridcolor=CORES['border']),
            xaxis=dict(title="", gridcolor=CORES['border']),
            legend=dict(title="", bgcolor=CORES['card_bg'])
        )
        
        # Feature importance do melhor modelo
        feature_importance_fig = None
        if best_model.get('feature_importances'):
            feature_labels = [data_manager.traduzir_coluna(f) for f in features]
            importance_df = pd.DataFrame({
                'feature': feature_labels,
                'importance': best_model['feature_importances']
            }).sort_values('importance', ascending=True)
            
            fig_importance = px.bar(importance_df, x='importance', y='feature', orientation='h',
                                     title=f"Importância das Características - {best_model['model']}",
                                     color='importance',
                                     color_continuous_scale=['#888888', CORES['accent']])
            fig_importance.update_layout(
                template='plotly_dark',
                paper_bgcolor=CORES['card_bg'],
                plot_bgcolor=CORES['card_bg'],
                font_color=CORES['text'],
                showlegend=False,
                height=400,
                xaxis=dict(title="Importância", gridcolor=CORES['border']),
                yaxis=dict(title="", gridcolor=CORES['border'])
            )
            feature_importance_fig = dcc.Graph(figure=fig_importance)
        
        layout = html.Div([
            # Melhor modelo
            html.Div([
                html.Div([
                    html.H3(best_model['model'], style={'color': CORES['accent'], 'marginBottom': '5px'}),
                    html.H1(f"{best_model['f1_score']:.1%}", style={'fontSize': '48px', 'marginBottom': '5px'}),
                    html.P("Melhor F1-Score", style={'color': CORES['text_secondary']})
                ], style={'textAlign': 'center', 'padding': '30px', 'borderBottom': f'1px solid {CORES["border"]}'})
            ]),
            
            # Gráfico de acurácia
            html.Div([
                html.H4("Comparação de Acurácia", style={'marginBottom': '20px', 'color': CORES['text']}),
                dcc.Graph(figure=fig_comparison, config={'displayModeBar': False})
            ], style={'padding': '20px', 'borderBottom': f'1px solid {CORES["border"]}'}),
            
            # Gráfico de métricas
            html.Div([
                html.H4("Métricas por Modelo", style={'marginBottom': '20px', 'color': CORES['text']}),
                dcc.Graph(figure=fig_metrics, config={'displayModeBar': False})
            ], style={'padding': '20px', 'borderBottom': f'1px solid {CORES["border"]}'}),
            
            # Feature importance
            html.Div([
                feature_importance_fig
            ], style={'padding': '20px'}) if feature_importance_fig else html.Div()
            
        ])
        
        return layout, store_data
        
    except Exception as e:
        import traceback
        print(f"Erro: {traceback.format_exc()}")
        return html.P(f"❌ Erro: {str(e)}", style={'color': CORES['danger']}), {}