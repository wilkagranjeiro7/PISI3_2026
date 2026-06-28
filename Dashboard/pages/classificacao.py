import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, callback, State
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, roc_curve
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from data_loader import data_manager
from model_manager import model_manager

# ==================================================
# TENTAR IMPORTAR MODELOS ADICIONAIS
# ==================================================

XGB_AVAILABLE = False
LGBM_AVAILABLE = False
CATBOOST_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    pass

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    pass

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    pass

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ==================================================
# CORES - VERSÃO SUAVE
# ==================================================

CORES = data_manager.get_cores()

TARGET_COLUMNS = ['day_strain', 'hrv', 'hrv_baseline']
LEAKAGE_FEATURES = {
    'day_strain', 'hrv', 'hrv_baseline',
    'strain_per_sleep', 'hrv_ratio', 'hrv_rhr_ratio'
}

METRIC_COLORS = [
    '#6C8EBF',  # Azul suave
    '#7CB3A1',  # Verde suave
    '#D4A574',  # Laranja suave
    '#B8A9C9',  # Roxo suave
    '#E8968C',  # Vermelho suave
    '#A8B5C0'   # Cinza suave
]

FAMILY_COLORS = {
    'Ensemble (Bagging)': '#6C8EBF',      
    'Ensemble (Boosting)': '#8B9DC3',     
    'Ensemble (Gradient Boosting)': '#A8B5C0',  
    'Linear': '#7CB3A1',                  
    'Tree': '#D4A574',                    
    'Instance-Based': '#E8968C'          
}

CORES_ROC = {
    'Random Forest': '#6C8EBF',
    'Gradient Boosting': '#8B9DC3',
    'XGBoost': '#B8A9C9',
    'LightGBM': '#D4A574',
    'CatBoost': '#E8968C',
    'Logistic Regression': '#7CB3A1',
    'Decision Tree': '#A8B5C0',
    'KNN': '#9CB8C9',
    'AdaBoost': '#C9B8A9'
}

# ==================================================
# CLASSIFICADORES
# ==================================================

CLASSIFIERS = {
    'Random Forest': {
        'model': RandomForestClassifier,
        'default': {'n_estimators': 200, 'random_state': 42, 'n_jobs': -1, 
                   'class_weight': 'balanced', 'max_depth': 15},
        'family': 'Ensemble (Bagging)',
        'shap_supported': True
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier,
        'default': {'n_estimators': 150, 'random_state': 42, 'subsample': 0.8, 
                   'learning_rate': 0.1, 'max_depth': 5},
        'family': 'Ensemble (Boosting)',
        'shap_supported': True
    },
    'Logistic Regression': {
        'model': LogisticRegression,
        'default': {'random_state': 42, 'max_iter': 1000, 'class_weight': 'balanced', 'C': 0.1},
        'family': 'Linear',
        'shap_supported': False
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier,
        'default': {'random_state': 42, 'class_weight': 'balanced', 'max_depth': 8},
        'family': 'Tree',
        'shap_supported': True
    },
    'KNN': {
        'model': KNeighborsClassifier,
        'default': {'n_neighbors': 15, 'weights': 'distance', 'p': 2},
        'family': 'Instance-Based',
        'shap_supported': False
    },
    'AdaBoost': {
        'model': AdaBoostClassifier,
        'default': {'n_estimators': 150, 'learning_rate': 0.1, 'random_state': 42},
        'family': 'Ensemble (Boosting)',
        'shap_supported': True
    },
}

if XGB_AVAILABLE:
    CLASSIFIERS['XGBoost'] = {
        'model': XGBClassifier,
        'default': {'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 6,
                   'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42,
                   'eval_metric': 'logloss', 'verbosity': 0},
        'family': 'Ensemble (Gradient Boosting)',
        'shap_supported': True
    }

if LGBM_AVAILABLE:
    CLASSIFIERS['LightGBM'] = {
        'model': LGBMClassifier,
        'default': {'n_estimators': 150, 'learning_rate': 0.1, 'num_leaves': 31,
                   'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42,
                   'verbose': -1},
        'family': 'Ensemble (Gradient Boosting)',
        'shap_supported': True
    }

if CATBOOST_AVAILABLE:
    CLASSIFIERS['CatBoost'] = {
        'model': CatBoostClassifier,
        'default': {'iterations': 150, 'learning_rate': 0.1, 'depth': 6,
                   'random_state': 42, 'verbose': False, 'auto_class_weights': 'Balanced'},
        'family': 'Ensemble (Gradient Boosting)',
        'shap_supported': True
    }

# ==================================================
# FUNÇÕES AUXILIARES
# ==================================================

def criar_features_derivadas(df):
    df_new = df.copy()
    if 'sleep_hours' in df.columns and 'sleep_efficiency' in df.columns:
        df_new['sleep_quality'] = df['sleep_hours'] * (df['sleep_efficiency'] / 100)
    if 'day_strain' in df.columns and 'sleep_hours' in df.columns:
        df_new['strain_per_sleep'] = df['day_strain'] / (df['sleep_hours'] + 0.1)
    if 'hrv' in df.columns and 'hrv_baseline' in df.columns:
        df_new['hrv_ratio'] = df['hrv'] / (df['hrv_baseline'] + 1)
    if 'resting_heart_rate' in df.columns and 'rhr_baseline' in df.columns:
        df_new['rhr_ratio'] = df['resting_heart_rate'] / (df['rhr_baseline'] + 1)
    if 'hrv' in df.columns and 'resting_heart_rate' in df.columns:
        df_new['hrv_rhr_ratio'] = df['hrv'] / (df['resting_heart_rate'] + 1)
    return df_new

def calcular_pontuacao_ponderada(result, pesos):
    pontuacao = 0
    total_peso = sum(pesos.values())
    for metrica, peso in pesos.items():
        if metrica in result and result[metrica] is not None:
            valor = result[metrica]
            pontuacao += (valor * peso) / total_peso
    return pontuacao

def criar_grafico_forca_shap(shap_values, feature_names, base_value, X_sample):
    if isinstance(shap_values, list):
        shap_vals = shap_values[0][0]
    else:
        shap_vals = shap_values[0] if len(shap_values.shape) > 2 else shap_values
        if len(shap_vals.shape) == 2:
            shap_vals = shap_vals[0]
            
    indices = np.argsort(np.abs(shap_vals))[::-1]
    feature_names_traduzidos = [data_manager.traduzir_coluna(f) for f in feature_names]
    
    df_shap = pd.DataFrame({
        'Feature': [feature_names_traduzidos[i] for i in indices],
        'SHAP Value': [shap_vals[i] for i in indices],
        'Feature Value': [X_sample.iloc[0][feature_names[i]] if hasattr(X_sample, 'iloc') else X_sample[0][i] for i in indices]
    })
    
    fig = go.Figure()
    colors = ['#E8968C' if x < 0 else '#7CB3A1' for x in df_shap['SHAP Value']]
    
    fig.add_trace(go.Bar(
        y=df_shap['Feature'],
        x=df_shap['SHAP Value'],
        orientation='h',
        marker_color=colors,
        text=[f"{v:.3f}" for v in df_shap['SHAP Value']],
        textposition='outside',
        name='Contribuição SHAP',
        hovertemplate='<b>%{y}</b><br>SHAP: %{x:.3f}<br>Valor: %{customdata:.2f}<extra></extra>',
        customdata=df_shap['Feature Value']
    ))
    
    fig.add_vline(x=base_value, line_dash="dash", line_color="#A8B5C0",
                  annotation_text=f"Base: {base_value:.3f}", annotation_position="top")
    
    valor_previsto = base_value + df_shap['SHAP Value'].sum()
    fig.add_vline(x=valor_previsto, line_color="#D4A574", line_width=2,
                  annotation_text=f"Previsto: {valor_previsto:.3f}", annotation_position="bottom")
    
    fig.update_layout(
        title="Gráfico de Força SHAP - Explicação da Previsão",
        template='plotly_dark',
        paper_bgcolor=CORES['card_bg'],
        plot_bgcolor=CORES['card_bg'],
        font_color=CORES['text'],
        height=max(400, len(feature_names) * 40),
        xaxis=dict(title="Contribuição para a Previsão", gridcolor=CORES['border'], zeroline=True, zerolinecolor=CORES['border']),
        yaxis=dict(title="", gridcolor=CORES['border'], autorange="reversed"),
        showlegend=False,
        margin=dict(l=150, r=50, t=50, b=50)
    )
    return fig

def criar_sumario_shap(shap_values, feature_names):
    if isinstance(shap_values, list):
        shap_vals = shap_values[0]
    else:
        shap_vals = shap_values
        
    feature_names_traduzidos = [data_manager.traduzir_coluna(f) for f in feature_names]
    mean_shap = np.abs(shap_vals).mean(axis=0)
    indices = np.argsort(mean_shap)[::-1][:10]
    
    df_summary = pd.DataFrame({
        'Feature': [feature_names_traduzidos[i] for i in indices],
        'Mean |SHAP|': [mean_shap[i] for i in indices]
    }).sort_values('Mean |SHAP|', ascending=True) # Horizontal ordena crescente de baixo pra cima
    
    fig = px.bar(
        df_summary,
        x='Mean |SHAP|',
        y='Feature',
        orientation='h',
        title="Importância SHAP - Média do Impacto Absoluto",
        color='Mean |SHAP|',
        color_continuous_scale=['#A8B5C0', '#6C8EBF'],
        text='Mean |SHAP|'
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=CORES['card_bg'],
        plot_bgcolor=CORES['card_bg'],
        font_color=CORES['text'],
        height=400,
        xaxis=dict(title="Impacto Médio", gridcolor=CORES['border']),
        yaxis=dict(title="", gridcolor=CORES['border']),
        showlegend=False
    )
    return fig

# ==================================================
# LAYOUT
# ==================================================

def create_layout(df):
    features_options = [
        {'label': data_manager.traduzir_coluna('resting_heart_rate'), 'value': 'resting_heart_rate'},
        {'label': data_manager.traduzir_coluna('sleep_hours'), 'value': 'sleep_hours'},
        {'label': data_manager.traduzir_coluna('sleep_efficiency'), 'value': 'sleep_efficiency'},
        {'label': 'Qualidade do Sono', 'value': 'sleep_quality'},
    ]
    
    models_options = [
        {'label': 'Random Forest', 'value': 'Random Forest'},
        {'label': 'Gradient Boosting', 'value': 'Gradient Boosting'},
    ]
    
    if XGB_AVAILABLE:
        models_options.append({'label': 'XGBoost', 'value': 'XGBoost'})
    if LGBM_AVAILABLE:
        models_options.append({'label': 'LightGBM', 'value': 'LightGBM'})
    if CATBOOST_AVAILABLE:
        models_options.append({'label': 'CatBoost', 'value': 'CatBoost'})
    
    models_options.extend([
        {'label': 'Regressão Logística', 'value': 'Logistic Regression'},
        {'label': 'Árvore de Decisão', 'value': 'Decision Tree'},
        {'label': 'KNN', 'value': 'KNN'},
        {'label': 'AdaBoost', 'value': 'AdaBoost'},
    ])
    
    return html.Div([
        html.Div([
            dbc.Button("← Voltar", href="/", color="light", size="sm",
                      style={'backgroundColor': 'transparent', 'border': f'1px solid {CORES["border"]}', 
                            'color': CORES['text']})
        ], style={'position': 'fixed', 'top': '20px', 'left': '20px', 'zIndex': '1000'}),
        
        html.Div([
            html.Div([
                html.H3("Classificação", style={'fontWeight': 'normal', 'marginBottom': '30px', 'color': CORES['text']}),
                html.P(
                    "Estimativa acadêmica de risco no mesmo dia, sem uso das variáveis que formam o alvo.",
                    style={'fontSize': '12px', 'color': CORES['text_secondary'],
                           'marginTop': '-25px', 'marginBottom': '25px'}
                ),
                
                html.Div([
                    html.Label("CARACTERÍSTICAS", style={'color': CORES['text_secondary'], 'fontSize': '12px', 'textTransform': 'uppercase'}),
                    dbc.Checklist(
                        id='class-features',
                        options=features_options,
                        value=['resting_heart_rate', 'sleep_hours', 'sleep_efficiency', 'sleep_quality'],
                        inline=False,
                        switch=True,
                        style={'marginTop': '10px'}
                    )
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.Label("MODELOS", style={'color': CORES['text_secondary'], 'fontSize': '12px', 'textTransform': 'uppercase'}),
                    dbc.Checklist(
                        id='class-models',
                        options=models_options,
                        value=['Random Forest'],
                        inline=False,
                        switch=True,
                        style={'marginTop': '10px'}
                    ),
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.Label("OPÇÕES DE AMBIENTE", style={'color': CORES['text_secondary'], 'fontSize': '12px', 'textTransform': 'uppercase'}),
                    dbc.Checkbox(
                        id='class-add-features',
                        label="Adicionar features derivadas",
                        value=True,
                        style={'marginTop': '10px'}
                    ),
                    dbc.Checkbox(
                        id='class-normalize',
                        label="Normalizar dados (Scaling)",
                        value=True,
                        style={'marginTop': '10px'}
                    ),
                    dbc.Checkbox(
                        id='class-shap',
                        label="Calcular SHAP",
                        value=False,
                        style={'marginTop': '10px'}
                    ),
                ], style={'marginBottom': '20px'}),
                
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
                'width': '340px', 
                'padding': '80px 20px 20px 20px',
                'borderRight': f'1px solid {CORES["border"]}',
                'height': '100vh',
                'overflowY': 'auto',
                'backgroundColor': CORES['background']
            }),
            
            html.Div([
                dcc.Store(id='class-results-store'),
                dcc.Store(id='class-model-store'),
                html.Div(id='class-results-container', children=[
                    html.Div([
                        html.P("Selecione os modelos preditivos e clique em Executar", 
                              style={'color': CORES['text_secondary'], 'textAlign': 'center', 'marginTop': '50px'})
                    ])
                ])
            ], style={'marginLeft': '360px', 'padding': '20px', 'minHeight': '100vh'})
        ])
    ], style={'backgroundColor': CORES['background'], 'minHeight': '100vh', 'color': CORES['text']})

# ==================================================
# CALLBACK PRINCIPAL
# ==================================================

@callback(
    [Output('class-results-container', 'children'),
     Output('class-results-store', 'data'),
     Output('class-model-store', 'data')],
    Input('class-run-button', 'n_clicks'),
    State('class-features', 'value'),
    State('class-models', 'value'),
    State('class-add-features', 'value'),
    State('class-normalize', 'value'),
    State('class-shap', 'value'),
    prevent_initial_call=True
)
def run_classification(n_clicks, features, selected_models, add_features, normalize, use_shap):
    if not selected_models:
        return html.P("Selecione pelo menos um modelo preditivo", style={'color': CORES['warning']}), {}, {}
    
    try:
        df = data_manager.get_clean_df()
        if df is None:
            return html.P("Base de dados indisponível no cache.", style={'color': CORES['danger']}), {}, {}
        
        # CORREÇÃO 1: FILTRAR OUTLIERS DO EXCEL ANTES DA ENGENHARIA DE FEATURES
        if 'sleep_hours' in df.columns:
            df = df[df['sleep_hours'] < 40000].copy()
            
        if add_features:
            df = criar_features_derivadas(df)
        
        features = features or []
        features_disponiveis = [
            feature for feature in features
            if feature in df.columns and feature not in LEAKAGE_FEATURES
        ]
        if not features_disponiveis:
            return html.P("Nenhuma feature selecionada está disponível no dataset", style={'color': CORES['danger']}), {}, {}
            
        # CORREÇÃO 2: VERIFICAÇÃO DE REQUISITOS PARA O TARGET DE OVERSTRAIN
        requisitos_target = TARGET_COLUMNS
        for col in requisitos_target:
            if col not in df.columns:
                return html.P(f"Coluna necessária para cálculo de Overstrain ausente: {col}", style={'color': CORES['danger']}), {}, {}
        
        # Filtragem de dados sem nulos
        colunas_finais = features_disponiveis + requisitos_target
        df_clean = df[list(dict.fromkeys(colunas_finais))].dropna().copy()
        
        if len(df_clean) < 100:
            return html.P(f"Volume de dados muito baixo após dropna: {len(df_clean)} linhas", style={'color': CORES['warning']}), {}, {}
            
        # CORREÇÃO 3: CRIAÇÃO DO TARGET ESTRATÉGICO DE OVERSTRAIN
        df_clean['target'] = ((df_clean['day_strain'] > df_clean['day_strain'].median()) & 
                              (df_clean['hrv'] < df_clean['hrv_baseline'])).astype(int)
        
        X = df_clean[features_disponiveis]
        y = df_clean['target']
        
        if len(y.unique()) < 2:
            return html.P("Classes desbalanceadas. Ajuste as features para recalcular a mediana.", style={'color': CORES['warning']}), {}, {}
            
        # Divisão estratificada estatisticamente robusta
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # CORREÇÃO 4: NORMALIZAÇÃO INDEPENDENTE 
        if normalize:
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features_disponiveis, index=X_train.index)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features_disponiveis, index=X_test.index)
        else:
            scaler = None
            X_train_scaled = X_train
            X_test_scaled = X_test
            
        results = []
        shap_models = {}
        trained_models = {}
        
        for model_name in selected_models:
            model_config = CLASSIFIERS.get(model_name)
            if model_config is None or model_config['model'] is None:
                continue
                
            print(f"Treinando {model_name}...")
            try:
                model = model_config['model'](**model_config['default'])
                model.fit(X_train_scaled, y_train)
                
                trained_models[model_name] = model
                
                if use_shap and SHAP_AVAILABLE and model_config.get('shap_supported', False):
                    shap_models[model_name] = model
                    
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_test_scaled)[:, 1]
                    y_pred = (y_prob >= 0.5).astype(int)
                    auc_score = roc_auc_score(y_test, y_prob)
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                else:
                    y_pred = model.predict(X_test_scaled)
                    y_prob = None
                    auc_score = None
                    fpr, tpr = None, None
                    
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                mcc = matthews_corrcoef(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                
                feature_importance = None
                if hasattr(model, 'feature_importances_'):
                    feature_importance = model.feature_importances_.tolist()
                elif hasattr(model, 'coef_') and len(model.coef_.shape) == 2:
                    feature_importance = np.abs(model.coef_[0]).tolist()
                    
                results.append({
                    'model': model_name,
                    'family': model_config.get('family', 'Desconhecido'),
                    'accuracy': float(accuracy),
                    'f1_score': float(f1),
                    'mcc': float(mcc),
                    'auc': float(auc_score) if auc_score is not None else None,
                    'fpr': fpr.tolist() if fpr is not None else None,
                    'tpr': tpr.tolist() if tpr is not None else None,
                    'confusion_matrix': cm.tolist(),
                    'feature_importance': feature_importance,
                })
            except Exception as e:
                print(f"Erro ao treinar {model_name}: {e}")
                continue
                
        if not results:
            return html.P("Nenhum algoritmo conseguiu concluir o treinamento.", style={'color': CORES['danger']}), {}, {}
            
        pesos = {'auc': 4, 'f1_score': 3, 'mcc': 2, 'accuracy': 1}
        for result in results:
            result['score'] = calcular_pontuacao_ponderada(result, pesos)
            
        results.sort(key=lambda x: x['score'], reverse=True)
        best_model = results[0]
        best_model_obj = trained_models.get(best_model['model'])
        
        # Salvar o modelo campeão no model_manager
        if best_model_obj is not None:
            try:
                metricas_para_salvar = {
                    'accuracy': best_model['accuracy'],
                    'f1_score': best_model['f1_score'],
                    'mcc': best_model['mcc'],
                    'auc': best_model['auc'],
                    'score': best_model['score'],
                    'feature_importance': best_model.get('feature_importance')
                }
                model_manager.salvar_modelo(
                    model=best_model_obj,
                    scaler=scaler,
                    features=features_disponiveis,
                    metrics=metricas_para_salvar,
                    nome_modelo=f"overstrain_{best_model['model'].lower().replace(' ', '_')}"
                )
                mensagem_salvamento = html.Div([
                    html.Span("✓ ", style={'color': CORES['success']}),
                    f"Modelo '{best_model['model']}' implantado em produção (saved_models/)"
                ], style={'color': CORES['success'], 'fontSize': '14px', 'marginTop': '10px'})
            except Exception as e:
                mensagem_salvamento = html.Div([f"⚠ Erro ao salvar artefatos: {str(e)}"], style={'color': CORES['danger']})
        else:
            mensagem_salvamento = html.Div(["⚠ Instância do modelo ausente"], style={'color': CORES['warning']})
            
        # ================================================
        # PROCESSAMENTO GRÁFICO (PLOTLY)
        # ================================================
        df_plot = pd.DataFrame(results)
        fig_accuracy = px.bar(
            df_plot, x='model', y='accuracy',
            title="Acurácia por Modelo (Alvo: Risco de Overstrain)",
            text=[f'{v:.1%}' for v in df_plot['accuracy']],
            color='family', color_discrete_map=FAMILY_COLORS
        )
        fig_accuracy.update_traces(textposition='outside')
        fig_accuracy.update_layout(
            template='plotly_dark', paper_bgcolor=CORES['card_bg'], plot_bgcolor=CORES['card_bg'],
            font_color=CORES['text'], height=400,
            yaxis=dict(title="Acurácia", tickformat='.0%', range=[0, 1], gridcolor=CORES['border']),
            xaxis=dict(title="", gridcolor=CORES['border'])
        )
        
        metrics_data = []
        for res in results:
            metrics_data.append({'Modelo': res['model'], 'Métrica': 'Acurácia', 'Valor': res['accuracy'] * 100})
            metrics_data.append({'Modelo': res['model'], 'Métrica': 'F1-Score', 'Valor': res['f1_score'] * 100})
            metrics_data.append({'Modelo': res['model'], 'Métrica': 'MCC', 'Valor': res['mcc'] * 100})
            metrics_data.append({'Modelo': res['model'], 'Métrica': 'Score Geral', 'Valor': res['score'] * 100})
            
        df_metrics = pd.DataFrame(metrics_data)
        fig_metrics = px.bar(
            df_metrics, x='Valor', y='Modelo', color='Métrica', barmode='group',
            title="Métricas de Desempenho (%)",
            color_discrete_map={
                'Acurácia': METRIC_COLORS[0], 'F1-Score': METRIC_COLORS[1],
                'MCC': METRIC_COLORS[2], 'Score Geral': METRIC_COLORS[3]
            }, text='Valor', orientation='h'
        )
        fig_metrics.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_metrics.update_layout(
            template='plotly_dark', paper_bgcolor=CORES['card_bg'], plot_bgcolor=CORES['card_bg'],
            font_color=CORES['text'], height=max(350, len(results) * 60),
            xaxis=dict(title="Porcentagem (%)", range=[0, 110], gridcolor=CORES['border']),
            yaxis=dict(title="", gridcolor=CORES['border'])
        )
        
        # CORREÇÃO 5: CONFIGURAÇÃO DE PROPORÇÃO DA CURVA ROC 
        fig_roc = None
        modelos_com_roc = [r for r in results if r.get('fpr') is not None and r.get('tpr') is not None]
        if modelos_com_roc:
            fig_roc = go.Figure()
            for res in modelos_com_roc:
                fig_roc.add_trace(go.Scatter(
                    x=[v * 100 for v in res['fpr']], y=[v * 100 for v in res['tpr']],
                    mode='lines', name=f"{res['model']} (AUC = {res['auc']:.3f})",
                    line=dict(color=CORES_ROC.get(res['model'], CORES['text_secondary']), width=2)
                ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 100], y=[0, 100], mode='lines', name='Linha de Baseline (Aleatório)',
                line=dict(dash='dash', color=CORES['text_secondary'], width=1)
            ))
            fig_roc.update_layout(
                title="Curva ROC - Capacidade Discriminante",
                template='plotly_dark', paper_bgcolor=CORES['card_bg'], plot_bgcolor=CORES['card_bg'],
                font_color=CORES['text'], height=450, width=450, # Garante área simétrica quadrada inicial
                xaxis=dict(title="Taxa de Falsos Positivos (%)", range=[0, 100], gridcolor=CORES['border']),
                yaxis=dict(title="Taxa de Verdadeiros Positivos (%)", range=[0, 100], gridcolor=CORES['border'],
                           scaleanchor="x", scaleratio=1), # Força a mesma proporção nos eixos x e y!
                legend=dict(bgcolor='rgba(0,0,0,0.6)', yanchor='bottom', y=0.02, xanchor='right', x=0.98)
            )
            
        fig_cm = None
        if best_model.get('confusion_matrix'):
            cm = np.array(best_model['confusion_matrix'], dtype=int)
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm, x=['Negativo (Normal)', 'Positivo (Overstrain)'], y=['Negativo (Normal)', 'Positivo (Overstrain)'],
                text=cm, texttemplate="%{text}", textfont={"size": 14, "color": CORES['text']},
                colorscale=[[0, '#6C8EBF'], [1, '#7CB3A1']], showscale=False
            ))
            fig_cm.update_layout(
                title=f"Matriz de Confusão - {best_model['model']}",
                template='plotly_dark', paper_bgcolor=CORES['card_bg'], plot_bgcolor=CORES['card_bg'],
                font_color=CORES['text'], height=350,
                xaxis=dict(title="Predito pelo Modelo", gridcolor=CORES['border']),
                yaxis=dict(title="Valor Real Biológico", gridcolor=CORES['border'])
            )
            
        # CORREÇÃO 6: ORDENAÇÃO DE CRESCENTE PARA DECRESCENTE DA FEATURE IMPORTANCE
        fig_importance = None
        if best_model.get('feature_importance') and features_disponiveis:
            importance = best_model['feature_importance']
            if len(importance) == len(features_disponiveis):
                feature_names_traduzidos = [data_manager.traduzir_coluna(f) for f in features_disponiveis]
                df_importance = pd.DataFrame({
                    'Feature': feature_names_traduzidos,
                    'Importance': importance
                }).sort_values('Importance', ascending=True) # Horizontal plota de baixo para cima, deixando o maior no topo
                
                fig_importance = px.bar(
                    df_importance, x='Importance', y='Feature', orientation='h',
                    title=f"Importância das Características - {best_model['model']}",
                    color='Importance', color_continuous_scale=['#A8B5C0', '#6C8EBF'], text='Importance'
                )
                fig_importance.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig_importance.update_layout(
                    template='plotly_dark', paper_bgcolor=CORES['card_bg'], plot_bgcolor=CORES['card_bg'],
                    font_color=CORES['text'], height=max(350, len(features_disponiveis) * 45),
                    xaxis=dict(title="Gini Importance / Coeficiente Absoluto", gridcolor=CORES['border']),
                    yaxis=dict(title="", gridcolor=CORES['border']), coloraxis_showscale=False
                )
                
        # ================================================
        # EXPANDIR SEÇÃO SHAP SE ATIVADA
        # ================================================
        shap_section = None
        if use_shap and SHAP_AVAILABLE and best_model['model'] in shap_models:
            try:
                model = shap_models[best_model['model']]
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_scaled)
                
                # Tratamento de segurança para dimensões do SHAP em classificações binárias
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values_input = shap_values[1]
                else:
                    shap_values_input = shap_values
                    
                shap_force_fig = criar_grafico_forca_shap(
                    shap_values_input, features_disponiveis,
                    explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
                    X_test_scaled.iloc[[0]]
                )
                shap_summary_fig = criar_sumario_shap(shap_values_input, features_disponiveis)
                
                shap_section = html.Div([
                    html.Hr(style={'borderColor': CORES['border']}),
                    html.H4("SHAP - Explicabilidade Clínica", style={'color': CORES['text']}),
                    dcc.Graph(figure=shap_force_fig, config={'displayModeBar': False}),
                    dcc.Graph(figure=shap_summary_fig, config={'displayModeBar': False})
                ])
            except Exception as e:
                shap_section = html.Div([html.P(f"Aviso SHAP: {str(e)}", style={'color': CORES['warning']})])
                
        # Métricas em blocos
        cm = np.array(best_model['confusion_matrix'], dtype=int)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
        sensibilidade = tp / (tp + fn) if (tp + fn) > 0 else 0
        especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0
        precisao = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        metrics_cards = html.Div([
            html.H4(f"Algoritmo Selecionado: {best_model['model']}", style={'color': CORES['accent']}),
            html.P("Abordagem: classificação acadêmica de risco no mesmo dia (alvo pela mediana)", style={'color': CORES['text_secondary']}),
            mensagem_salvamento,
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([html.H4(f"{best_model['accuracy']:.1%}", style={'color': '#7CB3A1'}), html.P("Acurácia", style={'fontSize':'12px'})]), style={'backgroundColor': CORES['card_bg']}), md=3),
                dbc.Col(dbc.Card(dbc.CardBody([html.H4(f"{best_model['f1_score']:.3f}", style={'color': '#6C8EBF'}), html.P("F1-Score", style={'fontSize':'12px'})]), style={'backgroundColor': CORES['card_bg']}), md=3),
                dbc.Col(dbc.Card(dbc.CardBody([html.H4(f"{best_model['mcc']:.3f}", style={'color': '#D4A574'}), html.P("MCC", style={'fontSize':'12px'})]), style={'backgroundColor': CORES['card_bg']}), md=3),
                dbc.Col(dbc.Card(dbc.CardBody([html.H4(f"{best_model.get('auc', 0):.3f}" if best_model.get('auc') else "N/A", style={'color': '#B8A9C9'}), html.P("AUC-ROC", style={'fontSize':'12px'})]), style={'backgroundColor': CORES['card_bg']}), md=3),
            ], className="mb-3 mt-3"),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([html.H4(f"{sensibilidade:.1%}", style={'color': '#7CB3A1'}), html.P("Sensibilidade (Recall)", style={'fontSize':'12px'})]), style={'backgroundColor': CORES['card_bg']}), md=4),
                dbc.Col(dbc.Card(dbc.CardBody([html.H4(f"{especificidade:.1%}", style={'color': '#A8B5C0'}), html.P("Especificidade", style={'fontSize':'12px'})]), style={'backgroundColor': CORES['card_bg']}), md=4),
                dbc.Col(dbc.Card(dbc.CardBody([html.H4(f"{precisao:.1%}", style={'color': '#6C8EBF'}), html.P("Precisão Preditiva", style={'fontSize':'12px'})]), style={'backgroundColor': CORES['card_bg']}), md=4),
            ], className="mb-3")
        ])
        
        # Estruturação final da tela
        layout = html.Div([
            metrics_cards,
            html.Hr(style={'borderColor': CORES['border']}),
            dbc.Row([
                dbc.Col([dcc.Graph(figure=fig_accuracy, config={'displayModeBar': False})], md=6),
                dbc.Col([dcc.Graph(figure=fig_metrics, config={'displayModeBar': False})], md=6)
            ]),
            html.Hr(style={'borderColor': CORES['border']}),
            dbc.Row([
                dbc.Col([dcc.Graph(figure=fig_roc, config={'displayModeBar': False})] if fig_roc else [], md=6, style={'display':'flex','justifyContent':'center'}),
                dbc.Col([dcc.Graph(figure=fig_cm, config={'displayModeBar': False})] if fig_cm else [], md=6)
            ]),
            html.Hr(style={'borderColor': CORES['border']}) if fig_importance else None,
            dcc.Graph(figure=fig_importance, config={'displayModeBar': False}) if fig_importance else None,
            shap_section,
            html.Hr(style={'borderColor': CORES['border']}),
            html.P(f"Dataset de Treino Ajustado Estritamente | Amostras Ativas: {len(df_clean)} registros limpos.",
                  style={'color': CORES['text_secondary'], 'textAlign': 'center', 'fontSize':'13px'})
        ])
        
        model_data = {
            'best_model_name': best_model['model'],
            'features': features_disponiveis,
            'metrics': best_model,
            'normalize': normalize
        }
        return layout, results, model_data
    except Exception as e:
        import traceback
        traceback.print_exc()
        return html.P(f"Falha de Execução do Pipeline: {str(e)}", style={'color': CORES['danger']}), {}, {}