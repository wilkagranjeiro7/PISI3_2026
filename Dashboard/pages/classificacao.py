# ==================================================
# pages/classificacao.py - VERSÃO COMPLETA OTIMIZADA
# ==================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, callback, State, dash
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, roc_curve, precision_score, recall_score
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
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
# CORES
# ==================================================

CORES = data_manager.get_cores()

# Paleta de cores suaves para métricas
METRIC_COLORS = [
    '#6C8EBF',  # Azul suave
    '#7CB3A1',  # Verde suave
    '#D4A574',  # Laranja suave
    '#B8A9C9',  # Roxo suave
    '#E8968C',  # Vermelho suave
    '#A8B5C0'   # Cinza suave
]

# Cores por família de modelos
FAMILY_COLORS = {
    'Ensemble (Bagging)': '#6C8EBF',
    'Ensemble (Boosting)': '#8B9DC3',
    'Ensemble (Gradient Boosting)': '#A8B5C0',
    'Linear': '#7CB3A1',
    'Tree': '#D4A574',
    'Instance-Based': '#E8968C',
}

# Cores para Curva ROC
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
# CLASSIFICADORES COM PARÂMETROS ANTI-OVERFITTING
# ==================================================

CLASSIFIERS = {
    'Random Forest': {
        'model': RandomForestClassifier,
        'default': {
            'n_estimators': 200,
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced',
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt'
        },
        'family': 'Ensemble (Bagging)',
        'shap_supported': True
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier,
        'default': {
            'n_estimators': 150,
            'random_state': 42,
            'subsample': 0.8,
            'learning_rate': 0.05,
            'max_depth': 4,
            'min_samples_split': 10,
            'min_samples_leaf': 5
        },
        'family': 'Ensemble (Boosting)',
        'shap_supported': True
    },
    'Logistic Regression': {
        'model': LogisticRegression,
        'default': {
            'random_state': 42,
            'max_iter': 1000,
            'class_weight': 'balanced',
            'C': 0.1,
            'penalty': 'l2'
        },
        'family': 'Linear',
        'shap_supported': True
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier,
        'default': {
            'random_state': 42,
            'class_weight': 'balanced',
            'max_depth': 5,
            'min_samples_split': 15,
            'min_samples_leaf': 10,
            'max_features': 'sqrt'
        },
        'family': 'Tree',
        'shap_supported': True
    },
    'KNN': {
        'model': KNeighborsClassifier,
        'default': {
            'n_neighbors': 20,
            'weights': 'distance',
            'p': 2
        },
        'family': 'Instance-Based',
        'shap_supported': False
    },
    'AdaBoost': {
        'model': AdaBoostClassifier,
        'default': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'random_state': 42
        },
        'family': 'Ensemble (Boosting)',
        'shap_supported': True
    },
}

if XGB_AVAILABLE:
    CLASSIFIERS['XGBoost'] = {
        'model': XGBClassifier,
        'default': {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 4,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'verbosity': 0,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'early_stopping_rounds': 50
        },
        'family': 'Ensemble (Gradient Boosting)',
        'shap_supported': True
    }

if LGBM_AVAILABLE:
    CLASSIFIERS['LightGBM'] = {
        'model': LGBMClassifier,
        'default': {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'num_leaves': 15,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_samples': 20
        },
        'family': 'Ensemble (Gradient Boosting)',
        'shap_supported': True
    }

if CATBOOST_AVAILABLE:
    CLASSIFIERS['CatBoost'] = {
        'model': CatBoostClassifier,
        'default': {
            'iterations': 200,
            'learning_rate': 0.05,
            'depth': 4,
            'random_state': 42,
            'verbose': False,
            'auto_class_weights': 'Balanced',
            'l2_leaf_reg': 3.0,
            'min_data_in_leaf': 10
        },
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

def encontrar_melhor_limiar(model, X_val, y_val):
    """Encontra o melhor limiar de classificação baseado no F1-Score"""
    y_proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.3, 0.8, 50)
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1

def detectar_overfitting(model, X_train, y_train, X_test, y_test):
    """Detecta se o modelo está overfitting"""
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    diff = train_acc - test_acc
    
    train_f1 = f1_score(y_train, model.predict(X_train))
    test_f1 = f1_score(y_test, model.predict(X_test))
    diff_f1 = train_f1 - test_f1
    
    if diff > 0.15:
        status = "🚨 ALTO OVERFITTING"
        cor = CORES['danger']
        icone = "🔴"
        mensagem = f"Diferença de {diff:.1%} entre treino e teste - O modelo está decorando os dados!"
        recomendacao = "Use mais dados, reduza complexidade, aumente regularização"
    elif diff > 0.08:
        status = "⚡ Overfitting Moderado"
        cor = CORES['warning']
        icone = "🟡"
        mensagem = f"Diferença de {diff:.1%} entre treino e teste - Cuidado com overfitting"
        recomendacao = "Considere reduzir complexidade ou usar cross-validation"
    else:
        status = "✅ Modelo Generaliza Bem"
        cor = CORES['success']
        icone = "🟢"
        mensagem = f"Diferença de {diff:.1%} entre treino e teste - Modelo consistente"
        recomendacao = "Modelo pronto para uso"
    
    return {
        'status': status,
        'cor': cor,
        'icone': icone,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'diff': diff,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'diff_f1': diff_f1,
        'mensagem': mensagem,
        'recomendacao': recomendacao
    }

# ==================================================
# FUNÇÕES PARA GRÁFICOS CORRIGIDOS
# ==================================================

def criar_grafico_acuracia(results):
    """Cria gráfico de acurácia com eixo Y de 0 a 1"""
    df_plot = pd.DataFrame(results)
    
    fig = px.bar(
        df_plot, 
        x='model', 
        y='accuracy',
        title="Acurácia por Modelo (Limiar ≥ 75)",
        text=[f'{v:.1%}' for v in df_plot['accuracy']],
        color='family',
        color_discrete_map=FAMILY_COLORS
    )
    fig.update_traces(textposition='outside', textfont=dict(size=12))
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=CORES['card_bg'],
        plot_bgcolor=CORES['card_bg'],
        font_color=CORES['text'],
        height=450,
        yaxis=dict(
            title="Acurácia",
            tickformat='.0%',
            range=[0, 1],
            gridcolor=CORES['border'],
            dtick=0.1
        ),
        xaxis=dict(title="Modelo", gridcolor=CORES['border']),
        legend=dict(
            title="Família", 
            bgcolor=CORES['card_bg'],
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        margin=dict(l=20, r=20, t=60, b=50)
    )
    return fig

def criar_grafico_roc(results):
    """Cria curva ROC com proporção 1:1"""
    modelos_com_roc = [r for r in results if r.get('fpr') is not None and r.get('tpr') is not None]
    
    if not modelos_com_roc:
        return None
    
    fig = go.Figure()
    
    for res in modelos_com_roc:
        fig.add_trace(go.Scatter(
            x=[v * 100 for v in res['fpr']],
            y=[v * 100 for v in res['tpr']],
            mode='lines',
            name=f"{res['model']} (AUC = {res['auc']:.3f})",
            line=dict(color=CORES_ROC.get(res['model'], CORES['text_secondary']), width=2)
        ))
    
    fig.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100],
        mode='lines',
        name='Classificador Aleatório',
        line=dict(dash='dash', color=CORES['text_secondary'], width=1)
    ))
    
    fig.update_layout(
        title="Curva ROC - Comparação entre Modelos",
        template='plotly_dark',
        paper_bgcolor=CORES['card_bg'],
        plot_bgcolor=CORES['card_bg'],
        font_color=CORES['text'],
        height=500,
        width=500,
        xaxis=dict(
            title="Taxa de Falsos Positivos (%)",
            range=[0, 100],
            gridcolor=CORES['border'],
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            title="Taxa de Verdadeiros Positivos (%)",
            range=[0, 100],
            gridcolor=CORES['border']
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0.6)',
            yanchor='bottom',
            y=0.02,
            xanchor='right',
            x=0.98
        ),
        margin=dict(l=20, r=20, t=60, b=50)
    )
    return fig

def criar_grafico_importancia(best_model, features_disponiveis):
    """Cria gráfico de importância ordenado do maior para o menor"""
    if not best_model.get('feature_importance'):
        return None
    
    importance = best_model['feature_importance']
    if len(importance) != len(features_disponiveis):
        return None
    
    feature_names_traduzidos = [data_manager.traduzir_coluna(f) for f in features_disponiveis]
    df_importance = pd.DataFrame({
        'Feature': feature_names_traduzidos,
        'Importance': importance
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(
        df_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f"Importância das Features",
        color='Importance',
        color_continuous_scale=['#A8B5C0', '#6C8EBF'],
        text='Importance'
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=CORES['card_bg'],
        plot_bgcolor=CORES['card_bg'],
        font_color=CORES['text'],
        height=max(400, len(features_disponiveis) * 35),
        xaxis=dict(
            title="Importância",
            gridcolor=CORES['border'],
            range=[0, max(df_importance['Importance']) * 1.1]
        ),
        yaxis=dict(
            title="",
            gridcolor=CORES['border'],
            autorange="reversed"
        ),
        showlegend=False,
        margin=dict(l=150, r=50, t=50, b=50)
    )
    return fig

def criar_matriz_confusao(best_model):
    """Cria matriz de confusão"""
    if not best_model.get('confusion_matrix'):
        return None
    
    cm = np.array(best_model['confusion_matrix'], dtype=int)
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Baixa', 'Alta'],
        y=['Baixa', 'Alta'],
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 14, "color": CORES['text']},
        colorscale=[[0, '#6C8EBF'], [1, '#7CB3A1']],
        showscale=True,
        colorbar=dict(title="Quantidade", tickfont=dict(color=CORES['text']))
    ))
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=CORES['card_bg'],
        plot_bgcolor=CORES['card_bg'],
        font_color=CORES['text'],
        height=400,
        xaxis=dict(title="Predito", gridcolor=CORES['border']),
        yaxis=dict(title="Real", gridcolor=CORES['border']),
        margin=dict(l=20, r=20, t=50, b=50)
    )
    return fig

def criar_grafico_metricas(results):
    """Cria gráfico de métricas por modelo"""
    metrics_data = []
    for res in results:
        metrics_data.append({'Modelo': res['model'], 'Métrica': 'Acurácia', 'Valor': res['accuracy'] * 100})
        metrics_data.append({'Modelo': res['model'], 'Métrica': 'F1-Score', 'Valor': res['f1_score'] * 100})
        metrics_data.append({'Modelo': res['model'], 'Métrica': 'MCC', 'Valor': res['mcc'] * 100})
        metrics_data.append({'Modelo': res['model'], 'Métrica': 'Score', 'Valor': res['score'] * 100})
    
    df_metrics = pd.DataFrame(metrics_data)
    fig = px.bar(
        df_metrics,
        x='Valor',
        y='Modelo',
        color='Métrica',
        barmode='group',
        title="Métricas por Modelo (%)",
        color_discrete_map={
            'Acurácia': METRIC_COLORS[0],
            'F1-Score': METRIC_COLORS[1],
            'MCC': METRIC_COLORS[2],
            'Score': METRIC_COLORS[3]
        },
        text='Valor',
        orientation='h'
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=CORES['card_bg'],
        plot_bgcolor=CORES['card_bg'],
        font_color=CORES['text'],
        height=max(350, len(results) * 50),
        xaxis=dict(title="Valor (%)", range=[0, 100], gridcolor=CORES['border']),
        yaxis=dict(title="", gridcolor=CORES['border']),
        legend=dict(
            title="",
            bgcolor=CORES['card_bg'],
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        margin=dict(l=20, r=20, t=60, b=50)
    )
    return fig

def plotar_curvas_aprendizado(model, X, y, cv=5):
    """Plota curvas de aprendizado"""
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y,
        cv=cv,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy',
        n_jobs=-1
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=train_sizes * 100,
        y=train_mean,
        mode='lines+markers',
        name='Treino',
        line=dict(color=CORES['accent'], width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=train_sizes * 100,
        y=test_mean,
        mode='lines+markers',
        name='Validação',
        line=dict(color=CORES['success'], width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes * 100, train_sizes[::-1] * 100]),
        y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(108, 142, 191, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes * 100, train_sizes[::-1] * 100]),
        y=np.concatenate([test_mean + test_std, (test_mean - test_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(124, 179, 161, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False
    ))
    
    fig.add_hline(y=0.75, line_dash="dash", line_color=CORES['text_secondary'],
                  annotation_text="Referência (75%)", annotation_position="bottom right")
    
    fig.update_layout(
        title="Curvas de Aprendizado - Diagnóstico de Overfitting",
        template='plotly_dark',
        paper_bgcolor=CORES['card_bg'],
        plot_bgcolor=CORES['card_bg'],
        font_color=CORES['text'],
        height=450,
        xaxis=dict(title="Tamanho do Conjunto de Treino (%)", gridcolor=CORES['border'], range=[0, 100]),
        yaxis=dict(title="Acurácia", gridcolor=CORES['border'], range=[0, 1]),
        legend=dict(bgcolor='rgba(0,0,0,0.5)', yanchor='bottom', y=0.02, xanchor='right', x=0.98),
        margin=dict(l=20, r=20, t=60, b=50)
    )
    
    # Diagnóstico
    diff_final = train_mean[-1] - test_mean[-1]
    if diff_final > 0.15:
        annotation_text = "⚠️ Overfitting Detectado!"
        annotation_color = CORES['danger']
    elif diff_final > 0.08:
        annotation_text = "⚡ Overfitting Moderado"
        annotation_color = CORES['warning']
    else:
        annotation_text = "✅ Modelo Generaliza Bem"
        annotation_color = CORES['success']
    
    fig.add_annotation(
        x=95,
        y=0.95,
        text=annotation_text,
        showarrow=False,
        font=dict(color=annotation_color, size=14, weight='bold'),
        bgcolor='rgba(0,0,0,0.7)',
        borderpad=10,
        borderwidth=2,
        bordercolor=annotation_color
    )
    
    return fig

# ==================================================
# FUNÇÕES PARA CLUSTERIZAÇÃO E REDUÇÃO DE DIMENSIONALIDADE
# ==================================================

def analisar_clusters(X, y, features):
    """Realiza análise de clusters e redução de dimensionalidade"""
    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    variancia_explicada = pca.explained_variance_ratio_
    
    # t-SNE (fallback se falhar)
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
        X_tsne = tsne.fit_transform(X)
    except:
        X_tsne = X_pca.copy()
    
    # K-Means
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    # Gráfico PCA
    fig_pca = px.scatter(
        x=X_pca[:, 0], y=X_pca[:, 1],
        color=y.astype(str),
        title=f"PCA - Redução de Dimensionalidade",
        labels={'x': f'PC1 ({variancia_explicada[0]:.1%})',
                'y': f'PC2 ({variancia_explicada[1]:.1%})'},
        color_discrete_sequence=['#E8968C', '#7CB3A1'],
        opacity=0.7
    )
    fig_pca.update_layout(
        template='plotly_dark',
        paper_bgcolor=CORES['card_bg'],
        plot_bgcolor=CORES['card_bg'],
        font_color=CORES['text'],
        height=450,
        xaxis=dict(gridcolor=CORES['border']),
        yaxis=dict(gridcolor=CORES['border']),
        margin=dict(l=20, r=20, t=60, b=50)
    )
    
    # Gráfico Clusters
    fig_clusters = px.scatter(
        x=X_pca[:, 0], y=X_pca[:, 1],
        color=clusters.astype(str),
        title="Clusterização dos Dados (K-Means)",
        labels={'x': 'PC1', 'y': 'PC2'},
        color_discrete_sequence=['#6C8EBF', '#D4A574'],
        opacity=0.7
    )
    fig_clusters.update_layout(
        template='plotly_dark',
        paper_bgcolor=CORES['card_bg'],
        plot_bgcolor=CORES['card_bg'],
        font_color=CORES['text'],
        height=450,
        xaxis=dict(gridcolor=CORES['border']),
        yaxis=dict(gridcolor=CORES['border']),
        margin=dict(l=20, r=20, t=60, b=50)
    )
    
    return {
        'pca': fig_pca,
        'clusters': fig_clusters,
        'variancia': variancia_explicada,
        'pca_components': pca.components_,
        'clusters_labels': clusters
    }

# ==================================================
# FUNÇÕES SHAP
# ==================================================

def criar_grafico_forca_shap(shap_values, feature_names, base_value, X_sample):
    if isinstance(shap_values, list):
        shap_vals = shap_values[0][0]
    else:
        shap_vals = shap_values[0]
    
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
        height=max(450, len(feature_names) * 40),
        xaxis=dict(title="Contribuição para a Previsão", gridcolor=CORES['border'],
                   zeroline=True, zerolinecolor=CORES['border']),
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
    })
    
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
        height=450,
        xaxis=dict(title="Impacto Médio", gridcolor=CORES['border']),
        yaxis=dict(title="", gridcolor=CORES['border']),
        showlegend=False,
        margin=dict(l=150, r=50, t=50, b=50)
    )
    return fig

# ==================================================
# LAYOUT
# ==================================================

def create_layout(df):
    features_options = [
        {'label': data_manager.traduzir_coluna('hrv'), 'value': 'hrv'},
        {'label': data_manager.traduzir_coluna('resting_heart_rate'), 'value': 'resting_heart_rate'},
        {'label': data_manager.traduzir_coluna('day_strain'), 'value': 'day_strain'},
        {'label': data_manager.traduzir_coluna('sleep_hours'), 'value': 'sleep_hours'},
        {'label': data_manager.traduzir_coluna('sleep_efficiency'), 'value': 'sleep_efficiency'},
        {'label': 'Qualidade do Sono', 'value': 'sleep_quality'},
        {'label': 'Strain por Hora de Sono', 'value': 'strain_per_sleep'},
        {'label': 'HRV/HR Ratio', 'value': 'hrv_rhr_ratio'},
        {'label': 'HRV Ratio', 'value': 'hrv_ratio'},
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
                
                html.Div([
                    html.Label("CARACTERÍSTICAS", style={'color': CORES['text_secondary'], 'fontSize': '12px', 'textTransform': 'uppercase'}),
                    dbc.Checklist(
                        id='class-features',
                        options=features_options,
                        value=['hrv', 'resting_heart_rate'],
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
                        value=['Random Forest', 'XGBoost'] if XGB_AVAILABLE else ['Random Forest'],
                        inline=False,
                        switch=True,
                        style={'marginTop': '10px'}
                    ),
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.Label("OPÇÕES", style={'color': CORES['text_secondary'], 'fontSize': '12px', 'textTransform': 'uppercase'}),
                    dbc.Checkbox(
                        id='class-add-features',
                        label="Adicionar features derivadas",
                        value=True,
                        style={'marginTop': '10px'}
                    ),
                    dbc.Checkbox(
                        id='class-normalize',
                        label="Normalizar dados",
                        value=True,
                        style={'marginTop': '10px'}
                    ),
                    dbc.Checkbox(
                        id='class-shap',
                        label="Calcular SHAP",
                        value=True,
                        style={'marginTop': '10px'}
                    ),
                    dbc.Checkbox(
                        id='class-cross-validation',
                        label="Usar Cross-Validation (5 folds)",
                        value=True,
                        style={'marginTop': '10px'}
                    ),
                    dbc.Checkbox(
                        id='class-cluster',
                        label="Análise de Clusters (PCA + K-Means)",
                        value=True,
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
                        html.P("Selecione os modelos e clique em Executar", 
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
    State('class-cross-validation', 'value'),
    State('class-cluster', 'value'),
    prevent_initial_call=True
)
def run_classification(n_clicks, features, selected_models, 
                       add_features, normalize, use_shap, use_cv, use_cluster):
    
    if not selected_models:
        return html.P("Selecione pelo menos um modelo", style={'color': CORES['warning']}), {}, {}
    
    try:
        # ================================================
        # CARREGAR DADOS
        # ================================================
        df = data_manager.get_clean_df()
        if df is None:
            return html.P("Dados não disponíveis.", style={'color': CORES['danger']}), {}, {}
        
        if add_features:
            df = criar_features_derivadas(df)
        
        features_disponiveis = [f for f in features if f in df.columns]
        if not features_disponiveis:
            return html.P("Nenhuma feature disponível", style={'color': CORES['danger']}), {}, {}
        
        # ================================================
        # PREPARAR DADOS
        # ================================================
        df_clean = df[features_disponiveis + ['recovery_score']].dropna().copy()
        
        if len(df_clean) < 100:
            return html.P(f"Dados insuficientes: {len(df_clean)}", style={'color': CORES['warning']}), {}, {}
        
        df_clean['target'] = (df_clean['recovery_score'] >= 75).astype(int)
        
        X = df_clean[features_disponiveis]
        y = df_clean['target']
        
        if len(y.unique()) < 2:
            return html.P("Todos os registros na mesma classe", style={'color': CORES['warning']}), {}, {}
        
        if normalize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_df = pd.DataFrame(X_scaled, columns=features_disponiveis)
        else:
            scaler = None
            X_df = X
        
        # ================================================
        # SEPARAR TREINO, VALIDAÇÃO E TESTE
        # ================================================
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_df, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        results = []
        shap_models = {}
        trained_models = {}
        overfitting_diagnostics = {}
        
        for model_name in selected_models:
            model_config = CLASSIFIERS.get(model_name)
            if model_config is None or model_config['model'] is None:
                continue
            
            print(f"Treinando {model_name}...")
            
            try:
                model = model_config['model'](**model_config['default'])
                model.fit(X_train, y_train)
                
                trained_models[model_name] = model
                print(f"Modelo {model_name} treinado")
                
                # ================================================
                # OTIMIZAR LIMIAR DE CLASSIFICAÇÃO
                # ================================================
                melhor_limiar, f1_otimizado = encontrar_melhor_limiar(model, X_val, y_val)
                print(f"Melhor limiar para {model_name}: {melhor_limiar:.2f}")
                
                # ================================================
                # DETECTAR OVERFITTING
                # ================================================
                overfit_diag = detectar_overfitting(model, X_train, y_train, X_test, y_test)
                overfitting_diagnostics[model_name] = overfit_diag
                
                # ================================================
                # CROSS-VALIDATION
                # ================================================
                cv_scores = None
                if use_cv:
                    try:
                        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                        cv_scores = cross_val_score(model, X_df, y, cv=cv, scoring='accuracy')
                        cv_mean = cv_scores.mean()
                        cv_std = cv_scores.std()
                        print(f"CV: {cv_mean:.3f} (+/- {cv_std:.3f})")
                    except Exception as e:
                        print(f"Erro no CV: {e}")
                        cv_scores = None
                
                if use_shap and SHAP_AVAILABLE and model_config.get('shap_supported', False):
                    shap_models[model_name] = model
                
                # Predições com o melhor limiar
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_test)[:, 1]
                    y_pred = (y_prob >= melhor_limiar).astype(int)
                    auc_score = roc_auc_score(y_test, y_prob)
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                else:
                    y_pred = model.predict(X_test)
                    y_prob = None
                    auc_score = None
                    fpr, tpr = None, None
                
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
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
                    'auc': float(auc_score) if auc_score else None,
                    'fpr': fpr.tolist() if fpr is not None else None,
                    'tpr': tpr.tolist() if tpr is not None else None,
                    'confusion_matrix': cm.tolist(),
                    'feature_importance': feature_importance,
                    'overfitting': overfit_diag,
                    'cv_scores': cv_scores.tolist() if cv_scores is not None else None,
                    'cv_mean': float(cv_scores.mean()) if cv_scores is not None else None,
                    'cv_std': float(cv_scores.std()) if cv_scores is not None else None,
                    'melhor_limiar': float(melhor_limiar),
                    'f1_otimizado': float(f1_otimizado)
                })
                
            except Exception as e:
                print(f"Erro {model_name}: {e}")
                continue
        
        if not results:
            return html.P("Nenhum modelo treinado!", style={'color': CORES['danger']}), {}, {}
        
        # ================================================
        # CALCULAR PONTUAÇÃO PONDERADA
        # ================================================
        pesos = {'auc': 4, 'f1_score': 3, 'mcc': 2, 'accuracy': 1}
        
        for result in results:
            result['score'] = calcular_pontuacao_ponderada(result, pesos)
            
            diff = result.get('overfitting', {}).get('diff', 0)
            if diff > 0.15:
                result['score'] *= 0.85
                result['score_penalizado'] = True
            elif diff > 0.08:
                result['score'] *= 0.95
                result['score_penalizado'] = True
            else:
                result['score_penalizado'] = False
        
        # ================================================
        # ORDENAR E SELECIONAR MELHOR MODELO
        # ================================================
        results.sort(key=lambda x: x['score'], reverse=True)
        best_model = results[0]
        best_model_obj = trained_models.get(best_model['model'])
        
        # ================================================
        # SALVAR MODELO COM REFERÊNCIAS
        # ================================================
        if best_model_obj is not None:
            try:
                print(f"Salvando modelo: {best_model['model']}")
                
                df_train_completo = X_train.copy()
                df_train_completo['recuperacao_alta'] = y_train.values
                
                df_alta = df_train_completo[df_train_completo['recuperacao_alta'] == 1]
                df_baixa = df_train_completo[df_train_completo['recuperacao_alta'] == 0]
                
                alta_recuperacao = {}
                baixa_recuperacao = {}
                
                for feature in features_disponiveis:
                    if feature in df_alta.columns and not df_alta.empty:
                        alta_recuperacao[feature] = float(df_alta[feature].mean())
                    else:
                        alta_recuperacao[feature] = 0.0
                        
                    if feature in df_baixa.columns and not df_baixa.empty:
                        baixa_recuperacao[feature] = float(df_baixa[feature].mean())
                    else:
                        baixa_recuperacao[feature] = 0.0
                
                percentis = {}
                for feature in features_disponiveis:
                    if feature in df_train_completo.columns:
                        percentis[feature] = {
                            'p25': float(df_train_completo[feature].quantile(0.25)),
                            'p50': float(df_train_completo[feature].quantile(0.50)),
                            'p75': float(df_train_completo[feature].quantile(0.75)),
                            'min': float(df_train_completo[feature].min()),
                            'max': float(df_train_completo[feature].max()),
                            'mean': float(df_train_completo[feature].mean()),
                            'std': float(df_train_completo[feature].std())
                        }
                    else:
                        percentis[feature] = {
                            'p25': 0.0,
                            'p50': 0.0,
                            'p75': 0.0,
                            'min': 0.0,
                            'max': 0.0,
                            'mean': 0.0,
                            'std': 0.0
                        }
                
                target_stats = {
                    'alta_recuperacao_count': int(df_alta.shape[0]) if not df_alta.empty else 0,
                    'baixa_recuperacao_count': int(df_baixa.shape[0]) if not df_baixa.empty else 0,
                    'total_count': int(df_train_completo.shape[0]),
                    'alta_percentual': float(df_alta.shape[0] / df_train_completo.shape[0]) if df_train_completo.shape[0] > 0 else 0
                }
                
                metricas_para_salvar = {
                    'accuracy': best_model['accuracy'],
                    'f1_score': best_model['f1_score'],
                    'mcc': best_model['mcc'],
                    'auc': best_model['auc'],
                    'score': best_model['score'],
                    'feature_importance': best_model.get('feature_importance', []),
                    'overfitting_diagnostic': best_model.get('overfitting', {}),
                    'cv_mean': best_model.get('cv_mean'),
                    'cv_std': best_model.get('cv_std'),
                    'melhor_limiar': best_model.get('melhor_limiar', 0.5),
                    'referencias': {
                        'alta_recuperacao': alta_recuperacao,
                        'baixa_recuperacao': baixa_recuperacao,
                        'percentis': percentis,
                        'target_stats': target_stats
                    },
                    'informacoes_treino': {
                        'n_amostras': int(X_train.shape[0]),
                        'n_features': int(X_train.shape[1]),
                        'features_utilizadas': features_disponiveis,
                        'normalizacao_utilizada': normalize,
                        'test_size': 0.3,
                        'val_size': 0.15,
                        'random_state': 42,
                        'cross_validation': use_cv
                    }
                }
                
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                nome_salvo = model_manager.salvar_modelo(
                    model=best_model_obj,
                    scaler=scaler,
                    features=features_disponiveis,
                    metrics=metricas_para_salvar,
                    nome_modelo=f"{best_model['model']}_{timestamp}"
                )
                
                print(f"✅ Modelo salvo: {nome_salvo}")
                
                overfit = best_model.get('overfitting', {})
                mensagem_salvamento = html.Div([
                    html.Span(overfit.get('icone', '✅'), style={'color': overfit.get('cor', CORES['success'])}),
                    html.Span(f" Modelo '{best_model['model']}' salvo com sucesso!", 
                             style={'color': overfit.get('cor', CORES['success'])}),
                    html.Br(),
                    html.Span(f"📊 {overfit.get('mensagem', '')}", 
                             style={'color': overfit.get('cor', CORES['text_secondary']), 'fontSize': '12px'}),
                    html.Br(),
                    html.Span(f"🎯 Limiar otimizado: {best_model.get('melhor_limiar', 0.5):.2f}", 
                             style={'color': CORES['accent'], 'fontSize': '12px'}),
                    html.Br(),
                    html.Span(f"💡 {overfit.get('recomendacao', '')}", 
                             style={'color': CORES['text_secondary'], 'fontSize': '12px'})
                ], style={'color': CORES['success'], 'fontSize': '14px', 'marginTop': '10px'})
                
            except Exception as e:
                print(f"❌ Erro ao salvar modelo: {e}")
                import traceback
                traceback.print_exc()
                mensagem_salvamento = html.Div([
                    html.Span("❌ ", style={'color': CORES['danger']}),
                    f" Erro ao salvar modelo: {str(e)}"
                ], style={'color': CORES['danger'], 'fontSize': '14px', 'marginTop': '10px'})
        else:
            mensagem_salvamento = html.Div([
                html.Span("⚠️ ", style={'color': CORES['warning']}),
                f" Modelo '{best_model['model']}' não encontrado para salvar"
            ], style={'color': CORES['warning'], 'fontSize': '14px', 'marginTop': '10px'})
        
        # ================================================
        # SHAP
        # ================================================
        shap_section = None
        if use_shap and SHAP_AVAILABLE and best_model['model'] in shap_models:
            try:
                model = shap_models[best_model['model']]
                if hasattr(model, 'predict_proba'):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_test)
                    
                    shap_force_fig = criar_grafico_forca_shap(
                        shap_values, features_disponiveis,
                        explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[0],
                        X_test.iloc[[0]]
                    )
                    shap_summary_fig = criar_sumario_shap(shap_values, features_disponiveis)
                    
                    shap_section = html.Div([
                        html.Hr(style={'borderColor': CORES['border'], 'marginTop': '30px', 'marginBottom': '30px'}),
                        html.H4("🔬 SHAP - Explicabilidade", style={'color': CORES['text'], 'marginBottom': '10px'}),
                        html.P("Como cada feature contribui para a previsão do modelo",
                              style={'color': CORES['text_secondary'], 'fontSize': '14px', 'marginBottom': '20px'}),
                        
                        html.H5("Gráfico de Força - Previsão Individual", style={'color': CORES['text'], 'marginTop': '20px'}),
                        dcc.Graph(figure=shap_force_fig, config={'displayModeBar': False}),
                        
                        html.H5("Resumo SHAP - Impacto Médio", style={'color': CORES['text'], 'marginTop': '30px'}),
                        dcc.Graph(figure=shap_summary_fig, config={'displayModeBar': False})
                    ])
                else:
                    shap_section = html.Div([
                        html.Hr(style={'borderColor': CORES['border'], 'marginTop': '30px'}),
                        html.H4("🔬 SHAP - Explicabilidade", style={'color': CORES['text']}),
                        html.P("Modelo não suporta SHAP", style={'color': CORES['warning']})
                    ])
            except Exception as e:
                shap_section = html.Div([
                    html.Hr(style={'borderColor': CORES['border'], 'marginTop': '30px'}),
                    html.H4("🔬 SHAP - Explicabilidade", style={'color': CORES['text']}),
                    html.P(f"Erro ao calcular SHAP: {str(e)}", style={'color': CORES['danger']})
                ])
        
        # ================================================
        # CLUSTERIZAÇÃO E REDUÇÃO DE DIMENSIONALIDADE
        # ================================================
        cluster_section = None
        if use_cluster:
            try:
                cluster_analysis = analisar_clusters(X_df, y, features_disponiveis)
                
                cluster_section = html.Div([
                    html.Hr(style={'borderColor': CORES['border'], 'marginTop': '30px', 'marginBottom': '30px'}),
                    html.H4("📊 Análise de Clusters e Redução de Dimensionalidade", 
                            style={'color': CORES['text'], 'marginBottom': '10px'}),
                    html.P("Visualização dos dados em 2D usando PCA",
                           style={'color': CORES['text_secondary'], 'fontSize': '14px', 'marginBottom': '20px'}),
                    
                    html.H5("PCA - Redução de Dimensionalidade", style={'color': CORES['text'], 'marginTop': '20px'}),
                    dcc.Graph(figure=cluster_analysis['pca'], config={'displayModeBar': False}),
                    
                    html.H5("Clusterização (K-Means)", style={'color': CORES['text'], 'marginTop': '30px'}),
                    dcc.Graph(figure=cluster_analysis['clusters'], config={'displayModeBar': False}),
                    
                    html.P(f"Variância explicada: PC1 = {cluster_analysis['variancia'][0]:.1%}, PC2 = {cluster_analysis['variancia'][1]:.1%}",
                           style={'color': CORES['text_secondary'], 'fontSize': '12px', 'textAlign': 'center', 'marginTop': '15px'})
                ])
            except Exception as e:
                cluster_section = html.Div([
                    html.Hr(style={'borderColor': CORES['border'], 'marginTop': '30px'}),
                    html.H4("📊 Análise de Clusters", style={'color': CORES['text']}),
                    html.P(f"Erro na análise de clusters: {str(e)}", style={'color': CORES['danger']})
                ])
        
        # ================================================
        # GRÁFICOS CORRIGIDOS
        # ================================================
        
        # 1. Acurácia com eixo Y de 0 a 1
        fig_accuracy = criar_grafico_acuracia(results)
        
        # 2. Métricas por modelo
        fig_metrics = criar_grafico_metricas(results)
        
        # 3. Curva ROC com proporção 1:1
        fig_roc = criar_grafico_roc(results)
        
        # 4. Importância (ordenada) 
        fig_importance = criar_grafico_importancia(best_model, features_disponiveis)
        
        # 5. Matriz de Confusão
        fig_cm = criar_matriz_confusao(best_model)
        
        # 6. Curvas de aprendizado
        learning_curve_fig = plotar_curvas_aprendizado(best_model_obj, X_df, y)
        
        # ================================================
        # CARDS DO MELHOR MODELO
        # ================================================
        best = best_model
        
        if best.get('confusion_matrix'):
            cm = np.array(best['confusion_matrix'], dtype=int)
            tn, fp, fn, tp = cm.ravel()
            sensibilidade = tp / (tp + fn) if (tp + fn) > 0 else 0
            especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0
            precisao = tp / (tp + fp) if (tp + fp) > 0 else 0
        else:
            sensibilidade = 0
            especificidade = 0
            precisao = 0
        
        overfit = best.get('overfitting', {})
        overfit_card = dbc.Card([
            dbc.CardBody([
                html.H5(f"{overfit.get('icone', '')} Diagnóstico de Overfitting", 
                       style={'color': overfit.get('cor', CORES['text'])}),
                html.P(overfit.get('mensagem', ''), 
                      style={'color': overfit.get('cor', CORES['text_secondary'])}),
                html.P(f"Treino: {overfit.get('train_acc', 0):.1%} | Teste: {overfit.get('test_acc', 0):.1%} | Diferença: {overfit.get('diff', 0):.1%}",
                      style={'color': CORES['text_secondary'], 'fontSize': '12px'}),
                html.P(f"🎯 Limiar otimizado: {best.get('melhor_limiar', 0.5):.2f}",
                      style={'color': CORES['accent'], 'fontSize': '12px'}),
                html.P(f"💡 {overfit.get('recomendacao', '')}",
                      style={'color': CORES['text_secondary'], 'fontSize': '12px', 'fontStyle': 'italic'})
            ])
        ], style={'backgroundColor': CORES['card_bg'], 'border': f'2px solid {overfit.get("cor", CORES["border"])}'})
        
        metrics_cards = html.Div([
            html.H4(f"🏆 Melhor Modelo: {best['model']}", style={'color': CORES['accent']}),
            html.P(f"Família: {best.get('family', 'Desconhecido')}", style={'color': CORES['text_secondary']}),
            html.P(f"Pontuação Ponderada: {best['score']:.4f}", style={'color': CORES['warning']}),
            html.P(f"Pesos: AUC=4, F1=3, MCC=2, Acurácia=1", 
                   style={'color': CORES['text_secondary'], 'fontSize': '12px'}),
            mensagem_salvamento,
            
            html.Hr(style={'borderColor': CORES['border'], 'marginTop': '20px', 'marginBottom': '20px'}),
            overfit_card,
            
            html.Hr(style={'borderColor': CORES['border'], 'marginTop': '25px', 'marginBottom': '25px'}),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H4(f"{best['accuracy']:.1%}", style={'color': '#7CB3A1'}),
                    html.P("Acurácia", style={'color': CORES['text_secondary'], 'fontSize': '12px'})
                ]), style={'backgroundColor': CORES['card_bg'], 'border': f'1px solid {CORES["border"]}'}), md=3),
                
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H4(f"{best['f1_score']:.3f}", style={'color': '#6C8EBF'}),
                    html.P("F1-Score", style={'color': CORES['text_secondary'], 'fontSize': '12px'})
                ]), style={'backgroundColor': CORES['card_bg'], 'border': f'1px solid {CORES["border"]}'}), md=3),
                
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H4(f"{best['mcc']:.3f}", style={'color': '#D4A574'}),
                    html.P("MCC", style={'color': CORES['text_secondary'], 'fontSize': '12px'})
                ]), style={'backgroundColor': CORES['card_bg'], 'border': f'1px solid {CORES["border"]}'}), md=3),
                
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H4(f"{best.get('auc', 0):.3f}" if best.get('auc') else "N/A", style={'color': '#B8A9C9'}),
                    html.P("AUC-ROC", style={'color': CORES['text_secondary'], 'fontSize': '12px'})
                ]), style={'backgroundColor': CORES['card_bg'], 'border': f'1px solid {CORES["border"]}'}), md=3),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H4(f"{sensibilidade:.1%}", style={'color': '#7CB3A1'}),
                    html.P("Sensibilidade", style={'color': CORES['text_secondary'], 'fontSize': '12px'})
                ]), style={'backgroundColor': CORES['card_bg'], 'border': f'1px solid {CORES["border"]}'}), md=3),
                
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H4(f"{especificidade:.1%}", style={'color': '#7CB3A1'}),
                    html.P("Especificidade", style={'color': CORES['text_secondary'], 'fontSize': '12px'})
                ]), style={'backgroundColor': CORES['card_bg'], 'border': f'1px solid {CORES["border"]}'}), md=3),
                
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H4(f"{precisao:.1%}", style={'color': '#6C8EBF'}),
                    html.P("Precisão", style={'color': CORES['text_secondary'], 'fontSize': '12px'})
                ]), style={'backgroundColor': CORES['card_bg'], 'border': f'1px solid {CORES["border"]}'}), md=3),
            ], className="mb-3")
        ])
        
        # ================================================
        # LAYOUT FINAL - UM GRÁFICO POR LINHA
        # ================================================
        layout = html.Div([
            
            # 1. Cards do Melhor Modelo
            metrics_cards,
            
            # 2. Curvas de Aprendizado
            html.Hr(style={'borderColor': CORES['border'], 'marginTop': '30px', 'marginBottom': '30px'}),
            html.H4("📈 Curvas de Aprendizado", style={'color': CORES['text'], 'marginBottom': '10px'}),
            html.P("Se as curvas estão próximas → modelo generaliza bem. Se treino está muito acima → overfitting.",
                  style={'color': CORES['text_secondary'], 'fontSize': '14px', 'marginBottom': '20px'}),
            dcc.Graph(figure=learning_curve_fig, config={'displayModeBar': False}),
            
            # 3. Acurácia
            html.Hr(style={'borderColor': CORES['border'], 'marginTop': '30px', 'marginBottom': '30px'}),
            html.H4("📊 Comparação de Acurácia", style={'color': CORES['text'], 'marginBottom': '10px'}),
            html.P("Acurácia de cada modelo no conjunto de teste",
                  style={'color': CORES['text_secondary'], 'fontSize': '14px', 'marginBottom': '20px'}),
            dcc.Graph(figure=fig_accuracy, config={'displayModeBar': False}),
            
            # 4. Métricas por Modelo
            html.Hr(style={'borderColor': CORES['border'], 'marginTop': '30px', 'marginBottom': '30px'}),
            html.H4("📊 Métricas por Modelo", style={'color': CORES['text'], 'marginBottom': '10px'}),
            html.P("Comparação de diferentes métricas para cada modelo",
                  style={'color': CORES['text_secondary'], 'fontSize': '14px', 'marginBottom': '20px'}),
            dcc.Graph(figure=fig_metrics, config={'displayModeBar': False}),
            
            # 5. Curva ROC
            html.Hr(style={'borderColor': CORES['border'], 'marginTop': '30px', 'marginBottom': '30px'}) if fig_roc else None,
            html.H4("📈 Curva ROC", style={'color': CORES['text'], 'marginBottom': '10px'}) if fig_roc else None,
            html.P("Taxa de Verdadeiros Positivos vs Taxa de Falsos Positivos",
                  style={'color': CORES['text_secondary'], 'fontSize': '14px', 'marginBottom': '20px'}) if fig_roc else None,
            dcc.Graph(figure=fig_roc, config={'displayModeBar': False}) if fig_roc else None,
            
            # 6. Importância das Features
            html.Hr(style={'borderColor': CORES['border'], 'marginTop': '30px', 'marginBottom': '30px'}),
            html.H4("🔍 Importância das Features", style={'color': CORES['text'], 'marginBottom': '10px'}),
            html.P("Quanto maior o valor, maior o impacto da feature no modelo",
                  style={'color': CORES['text_secondary'], 'fontSize': '14px', 'marginBottom': '20px'}),
            dcc.Graph(figure=fig_importance, config={'displayModeBar': False}) if fig_importance else html.P("Importância não disponível"),
            
            # 7. Matriz de Confusão
            html.Hr(style={'borderColor': CORES['border'], 'marginTop': '30px', 'marginBottom': '30px'}),
            html.H4("📋 Matriz de Confusão", style={'color': CORES['text'], 'marginBottom': '10px'}),
            html.P("Acertos e erros do modelo na classificação",
                  style={'color': CORES['text_secondary'], 'fontSize': '14px', 'marginBottom': '20px'}),
            dcc.Graph(figure=fig_cm, config={'displayModeBar': False}) if fig_cm else html.P("Matriz não disponível"),
            
            # 8. Clusters
            cluster_section,
            
            # 9. SHAP
            shap_section,
            
            # Rodapé
            html.Hr(style={'borderColor': CORES['border'], 'marginTop': '30px', 'marginBottom': '30px'}),
            html.P(f"{len(results)} modelos testados | Melhor: {best_model['model']} | Limiar: ≥ {best_model.get('melhor_limiar', 0.5):.2f}",
                  style={'color': CORES['text_secondary'], 'textAlign': 'center', 'fontSize': '12px'})
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
        return html.P(f"Erro: {str(e)}", style={'color': CORES['danger']}), {}, {}