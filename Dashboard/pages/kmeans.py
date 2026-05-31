# pages/kmeans.py
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# =========================================================
# CORES PADRÃO (via DataManager)
# =========================================================

from data_loader import data_manager
CORES = data_manager.get_cores()

# Cores para os clusters (baseado na paleta padrão)
CLUSTER_COLORS = [CORES['accent'], CORES['success'], CORES['warning'], CORES['hrv'], CORES['danger'], CORES['text_secondary']]


# =========================================================
# IDENTIFICAR COLUNA DE USUÁRIO
# =========================================================

def identificar_coluna_usuario(df):
    possiveis = ['user_id', 'UserId', 'UserID', 'user', 'athlete_id', 'USER_00001', 'USER']
    for col in possiveis:
        if col in df.columns:
            return col
    for col in df.columns:
        if 'user' in col.lower():
            return col
    return None


# =========================================================
# CARREGAR DADOS
# =========================================================

def get_user_data():
    print("\n" + "="*50)
    print("CARREGANDO DADOS PARA CLUSTERIZAÇÃO")
    print("="*50)
    
    try:
        from data_loader import data_manager
        
        df = data_manager.get_clean_df()
        
        if df is None:
            data_manager.load_data()
            df = data_manager.get_clean_df()
        
        if df is None:
            print("❌ Falha ao carregar dados")
            return None, None, None
        
        print(f"✅ {len(df):,} registros carregados")
        
        # Estatísticas do Recovery Score
        if 'recovery_score' in df.columns:
            print(f"\nScore de Recuperação:")
            print(f"   Média: {df['recovery_score'].mean():.2f}")
            print(f"   Mínimo: {df['recovery_score'].min():.2f}")
            print(f"   Máximo: {df['recovery_score'].max():.2f}")
        
        # Identificar coluna de usuário
        user_col = identificar_coluna_usuario(df)
        
        if user_col is None:
            print("❌ Coluna de usuário não encontrada")
            return None, None, None
        
        print(f"\n✅ Coluna de usuário: '{user_col}'")
        
        # Features para clusterização
        features = ['recovery_score', 'sleep_hours', 'day_strain', 'hrv', 'age', 'resting_heart_rate']
        features_existentes = [f for f in features if f in df.columns]
        
        print(f"✅ Features: {features_existentes}")
        
        # Agrupar por usuário (média)
        df_user = df.groupby(user_col)[features_existentes].mean().reset_index()
        df_user = df_user.dropna()
        
        print(f"✅ {len(df_user)} usuários únicos")
        
        return df_user, user_col, features_existentes
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return None, None, None


# =========================================================
# NOMEAR CLUSTERS
# =========================================================

def nomear_clusters(df, features):
    nomes = {}
    grouped = df.groupby('cluster')[features].mean()
    
    print("\nPerfil dos Clusters:")
    print("-" * 40)
    
    for cluster_id, row in grouped.iterrows():
        nome = f"Segmento {cluster_id + 1}"
        
        if 'recovery_score' in features:
            rec = row['recovery_score']
            print(f"\n   Cluster {cluster_id}: Score: {rec:.1f}")
            
            if rec >= 80:
                nome = "Performance Máxima"
            elif rec >= 70:
                nome = "Alta Performance"
            elif rec >= 60:
                nome = "Recuperação Adequada"
            elif rec >= 50:
                nome = "Recuperação Moderada"
            elif rec >= 40:
                nome = "Recuperação Baixa"
            else:
                nome = "Recuperação Crítica"
        
        nomes[cluster_id] = nome
    
    return nomes


# =========================================================
# EXECUTAR KMEANS
# =========================================================

def rodar_kmeans(k=3):
    print(f"\n{'='*50}")
    print(f"EXECUTANDO K-MEANS COM K={k}")
    print(f"{'='*50}")
    
    df_user, user_col, features = get_user_data()
    
    if df_user is None or user_col is None or not features:
        print("❌ Falha ao carregar dados")
        return None
    
    if len(features) < 2:
        print(f"❌ Features insuficientes")
        return None
    
    # Dar peso extra para recovery_score
    df_pesado = df_user.copy()
    if 'recovery_score' in df_pesado.columns:
        print(f"\nAplicando peso extra para Score de Recuperação (fator 2x)")
        df_pesado['recovery_score'] = df_pesado['recovery_score'] * 2
    
    X = df_pesado[features].values
    
    # Padronizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA para visualização
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # K-Means
    model = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10, batch_size=256)
    clusters = model.fit_predict(X_scaled)
    
    # Silhouette Score
    sil_score = silhouette_score(X_scaled, clusters) if len(set(clusters)) > 1 else 0
    
    # Importância das features
    if len(set(clusters)) > 1:
        f_scores, p_values = f_classif(X_scaled, clusters)
    else:
        f_scores = np.zeros(len(features))
        p_values = np.ones(len(features))
    
    importancia = sorted([(features[i], f_scores[i]) for i in range(len(features))], 
                        key=lambda x: x[1], reverse=True)
    
    # Centróides
    centroids = pca.transform(model.cluster_centers_)
    
    # DataFrame resultado
    df_resultado = df_user.copy()
    df_resultado['cluster'] = clusters
    df_resultado['x'] = X_pca[:, 0]
    df_resultado['y'] = X_pca[:, 1]
    
    # Nomear clusters
    nomes = nomear_clusters(df_resultado, features)
    df_resultado['perfil'] = df_resultado['cluster'].map(nomes)
    
    print(f"\n{'='*50}")
    print("RESULTADOS DA CLUSTERIZAÇÃO")
    print(f"{'='*50}")
    print(f"✅ Silhouette Score: {sil_score:.3f}")
    
    print(f"\nDistribuição dos usuários:")
    for cluster_id, count in df_resultado['cluster'].value_counts().sort_index().items():
        nome = nomes[cluster_id]
        rec_medio = df_resultado[df_resultado['cluster'] == cluster_id]['recovery_score'].mean()
        print(f"   {nome}: {count} usuários - Score médio: {rec_medio:.1f}")
    
    return {
        'df': df_resultado,
        'features': features,
        'importancia': importancia,
        'centroids': centroids,
        'silhouette': sil_score,
        'total_usuarios': len(df_resultado),
        'k': k,
        'X_scaled': X_scaled,
        'clusters': clusters
    }


# =========================================================
# GRÁFICOS
# =========================================================

def grafico_clusters(df, centroids):
    fig = go.Figure()
    
    for idx, cluster in enumerate(sorted(df['cluster'].unique())):
        dados = df[df['cluster'] == cluster]
        nome = dados['perfil'].iloc[0]
        rec_medio = dados['recovery_score'].mean()
        
        fig.add_trace(go.Scatter(
            x=dados['x'],
            y=dados['y'],
            mode='markers',
            name=nome,
            marker=dict(
                size=10,
                color=CLUSTER_COLORS[idx % len(CLUSTER_COLORS)],
                opacity=0.6,
                line=dict(width=1, color='white')
            ),
            hovertemplate=f'<b>{nome}</b><br>Recuperação: {rec_medio:.0f}<br>Usuários: {len(dados)}<extra></extra>'
        ))
    
    fig.add_trace(go.Scatter(
        x=centroids[:, 0],
        y=centroids[:, 1],
        mode='markers',
        marker=dict(size=15, color='white', symbol='x', line=dict(width=2, color=CORES['accent'])),
        name='Centroides',
        hovertemplate='<b>Centroide</b><extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Segmentação por Score de Recuperação - {len(df)} usuários",
        height=500,
        paper_bgcolor=CORES['card_bg'],
        plot_bgcolor=CORES['card_bg'],
        font=dict(color=CORES['text']),
        xaxis=dict(gridcolor=CORES['border'], zeroline=False, title='Componente Principal 1'),
        yaxis=dict(gridcolor=CORES['border'], zeroline=False, title='Componente Principal 2'),
        legend=dict(
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(0,0,0,0.6)',
            bordercolor=CORES['border'],
            borderwidth=1,
            font=dict(size=10)
        ),
        margin=dict(l=40, r=180, t=60, b=40)
    )
    
    return fig


def grafico_importancia(importancia):
    fig = go.Figure()
    
    nomes = [f[0].replace('_', ' ').title() for f in importancia[:8]]
    valores = [f[1] for f in importancia[:8]]
    
    fig.add_trace(go.Bar(
        x=valores,
        y=nomes,
        orientation='h',
        marker_color=CORES['accent'],
        text=[f'{v:.1f}' for v in valores],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Características Mais Importantes",
        xaxis_title="Pontuação F",
        yaxis_title="",
        height=400,
        paper_bgcolor=CORES['card_bg'],
        plot_bgcolor=CORES['card_bg'],
        font=dict(color=CORES['text'])
    )
    
    return fig


def grafico_cotovelo(df, features):
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    wcss = []
    k_range = range(2, min(9, len(df)))
    
    for k in k_range:
        km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(k_range), y=wcss, mode='lines+markers',
        marker=dict(size=10, color=CORES['accent']),
        line=dict(width=2, color=CORES['accent'])
    ))
    
    fig.update_layout(
        title="Método do Cotovelo",
        xaxis_title="Número de Segmentos (K)",
        yaxis_title="Inércia (WCSS)",
        height=400,
        paper_bgcolor=CORES['card_bg'],
        plot_bgcolor=CORES['card_bg'],
        font=dict(color=CORES['text'])
    )
    
    return fig


def grafico_silhueta(X_scaled, clusters, k):
    if len(set(clusters)) < 2:
        fig = go.Figure()
        fig.update_layout(title="Não é possível calcular Silhueta", paper_bgcolor=CORES['card_bg'])
        return fig
    
    silhouette_vals = silhouette_samples(X_scaled, clusters)
    
    fig = go.Figure()
    y_lower = 10
    
    for i in range(k):
        if i not in clusters:
            continue
        ith_cluster = silhouette_vals[clusters == i]
        ith_cluster.sort()
        y_upper = y_lower + len(ith_cluster)
        
        fig.add_trace(go.Bar(
            x=ith_cluster,
            y=list(range(y_lower, y_upper)),
            orientation='h',
            name=f'Cluster {i}',
            marker_color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
            opacity=0.7
        ))
        y_lower = y_upper + 10
    
    sil_avg = silhouette_score(X_scaled, clusters)
    fig.add_vline(x=sil_avg, line_dash="dash", line_color=CORES['warning'], 
                  annotation_text=f'Média: {sil_avg:.3f}')
    
    fig.update_layout(title=f"Silhueta (K={k})", xaxis_title="Coeficiente", height=400,
                      paper_bgcolor=CORES['card_bg'], plot_bgcolor=CORES['card_bg'],
                      font=dict(color=CORES['text']))
    
    return fig


def grafico_perfil_clusters(df, features):
    perfil = df.groupby('cluster')[features].mean()
    tamanhos = df['cluster'].value_counts()
    
    traducao_features = {
        'recovery_score': 'Score de Recuperação',
        'sleep_hours': 'Horas de Sono',
        'day_strain': 'Carga Diária',
        'hrv': 'HRV',
        'age': 'Idade',
        'resting_heart_rate': 'FC Repouso'
    }
    
    nomes_features = [traducao_features.get(f, f.replace('_', ' ').title()) for f in features[:6]]
    
    fig = go.Figure()
    
    for idx, cluster_id in enumerate(sorted(perfil.index)):
        valores = [perfil.loc[cluster_id, f] for f in features[:6]]
        nome_cluster = df[df['cluster'] == cluster_id]['perfil'].iloc[0]
        
        fig.add_trace(go.Bar(
            x=valores,
            y=nomes_features,
            orientation='h',
            name=f"{nome_cluster} ({tamanhos[cluster_id]} usuários)",
            marker_color=CLUSTER_COLORS[idx % len(CLUSTER_COLORS)],
            text=[f'{v:.1f}' for v in valores],
            textposition='outside',
            textfont=dict(size=10),
            hovertemplate='<b>%{y}</b><br>Valor: %{x:.1f}<br>%{fullData.name}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Perfil Comparativo dos Segmentos",
        xaxis_title="Valor Médio",
        yaxis_title="",
        height=500,
        barmode='group',
        paper_bgcolor=CORES['card_bg'],
        plot_bgcolor=CORES['card_bg'],
        font=dict(color=CORES['text']),
        legend=dict(
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(0,0,0,0.6)',
            bordercolor=CORES['border'],
            borderwidth=1,
            font=dict(size=10)
        ),
        margin=dict(l=150, r=180, t=60, b=50)
    )
    
    fig.update_xaxes(gridcolor=CORES['border'], zerolinecolor=CORES['border'])
    fig.update_yaxes(gridcolor=CORES['border'], zerolinecolor=CORES['border'])
    
    return fig


def grafico_radar(df, features):
    """Radar chart comparando os perfis dos clusters"""
    
    # Calcular médias por cluster
    perfil = df.groupby('cluster')[features].mean()
    tamanhos = df['cluster'].value_counts()
    nomes_clusters = df[['cluster', 'perfil']].drop_duplicates().set_index('cluster')['perfil'].to_dict()
    
    # Normalizar os dados para o radar (0-1)
    scaler = MinMaxScaler()
    perfil_normalizado = pd.DataFrame(
        scaler.fit_transform(perfil),
        columns=perfil.columns,
        index=perfil.index
    )
    
    # Traduzir nomes das features
    traducao_features = {
        'recovery_score': 'Score de Recuperação',
        'sleep_hours': 'Horas de Sono',
        'day_strain': 'Carga Diária',
        'hrv': 'HRV',
        'age': 'Idade',
        'resting_heart_rate': 'FC Repouso'
    }
    
    nomes_features = [traducao_features.get(f, f.replace('_', ' ').title()) for f in features[:6]]
    
    # Criar radar chart
    fig = go.Figure()
    
    for idx, cluster_id in enumerate(perfil.index):
        valores = perfil_normalizado.loc[cluster_id, features[:6]].values.tolist()
        # Fechar o polígono (repetir o primeiro valor)
        valores += valores[:1]
        nomes_fechados = nomes_features + [nomes_features[0]]
        
        nome_cluster = nomes_clusters.get(cluster_id, f"Cluster {cluster_id}")
        tamanho = tamanhos.get(cluster_id, 0)
        
        fig.add_trace(go.Scatterpolar(
            r=valores,
            theta=nomes_fechados,
            fill='toself',
            name=f"{nome_cluster} ({tamanho} usuários)",
            line=dict(color=CLUSTER_COLORS[idx % len(CLUSTER_COLORS)], width=2),
            fillcolor=CLUSTER_COLORS[idx % len(CLUSTER_COLORS)],
            opacity=0.3
        ))
    
    fig.update_layout(
        title="Perfil dos Segmentos (Radar)",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(color=CORES['text_secondary']),
                gridcolor=CORES['border']
            ),
            angularaxis=dict(
                tickfont=dict(color=CORES['text'], size=10),
                gridcolor=CORES['border']
            ),
            bgcolor=CORES['card_bg']
        ),
        height=500,
        paper_bgcolor=CORES['card_bg'],
        plot_bgcolor=CORES['card_bg'],
        font=dict(color=CORES['text']),
        legend=dict(
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(0,0,0,0.6)',
            bordercolor=CORES['border'],
            borderwidth=1,
            font=dict(size=10)
        ),
        margin=dict(l=80, r=180, t=80, b=80)
    )
    
    return fig


# =========================================================
# LAYOUT
# =========================================================

def create_layout(df=None):
    return html.Div([
        
        # Botão voltar (fixo no canto superior esquerdo)
        html.Div([
            dbc.Button(
                "← Voltar",
                href="/",
                color="light",
                size="sm",
                style={'backgroundColor': 'transparent', 'border': f'1px solid {CORES["border"]}', 
                       'color': CORES['text']}
            )
        ], style={'position': 'fixed', 'top': '20px', 'left': '20px', 'zIndex': '1000'}),
        
        # Conteúdo principal
        html.Div([
            # Painel esquerdo - configurações (FIXO)
            html.Div([
                html.H3("Segmentação", style={'fontWeight': 'normal', 'marginBottom': '30px', 'color': CORES['text']}),
                
                html.Div([
                    html.Label("NÚMERO DE SEGMENTOS (K)", style={'color': CORES['text_secondary'], 'fontSize': '12px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                    html.Div([
                        dcc.Slider(
                            id='k-slider',
                            min=2, max=6, step=1, value=3,
                            marks={i: {"label": str(i), "style": {"color": CORES['text_secondary']}} for i in range(2, 7)}
                        )
                    ], style={'marginTop': '10px'})
                ], style={'marginBottom': '30px'}),
                
                dbc.Button(
                    "Executar Segmentação", 
                    id='run-button', 
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
                html.Div(id='results-container', children=[
                    html.Div([
                        html.P("Selecione o número de segmentos e clique em Executar", 
                              style={'color': CORES['text_secondary'], 'textAlign': 'center', 'marginTop': '50px'})
                    ])
                ], style={'display': 'block'}),
                
                html.Div(id='results-graphs', style={'display': 'none'}, children=[
                    dbc.Row([
                        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='graph-clusters')), 
                                        style={"backgroundColor": CORES['card_bg'], "border": f"1px solid {CORES['border']}"}), 
                               width=12, className="mb-4"),
                        
                        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='graph-importancia')), 
                                        style={"backgroundColor": CORES['card_bg'], "border": f"1px solid {CORES['border']}"}), 
                               width=6, className="mb-4"),
                        
                        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='graph-cotovelo')), 
                                        style={"backgroundColor": CORES['card_bg'], "border": f"1px solid {CORES['border']}"}), 
                               width=6, className="mb-4"),
                        
                        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='graph-silhueta')), 
                                        style={"backgroundColor": CORES['card_bg'], "border": f"1px solid {CORES['border']}"}), 
                               width=12, className="mb-4"),
                        
                        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='graph-perfil')), 
                                        style={"backgroundColor": CORES['card_bg'], "border": f"1px solid {CORES['border']}"}), 
                               width=12, className="mb-4"),
                        
                        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='graph-radar')), 
                                        style={"backgroundColor": CORES['card_bg'], "border": f"1px solid {CORES['border']}"}), 
                               width=12, className="mb-4"),
                    ])
                ])
            ], style={'marginLeft': '320px', 'padding': '20px', 'minHeight': '100vh'})
            
        ])
        
    ], style={'backgroundColor': CORES['background'], 'minHeight': '100vh', 'color': CORES['text']})


# =========================================================
# CALLBACK
# =========================================================

@callback(
    [Output('results-graphs', 'style'),
     Output('graph-clusters', 'figure'),
     Output('graph-importancia', 'figure'),
     Output('graph-cotovelo', 'figure'),
     Output('graph-silhueta', 'figure'),
     Output('graph-perfil', 'figure'),
     Output('graph-radar', 'figure')],
    Input('run-button', 'n_clicks'),
    State('k-slider', 'value'),
    prevent_initial_call=True
)
def executar_analise(n_clicks, k):
    if n_clicks is None:
        raise PreventUpdate
    
    resultado = rodar_kmeans(k=k)
    
    if resultado is None:
        fig_vazia = go.Figure()
        fig_vazia.update_layout(paper_bgcolor=CORES['card_bg'], plot_bgcolor=CORES['card_bg'])
        return {'display': 'block'}, fig_vazia, fig_vazia, fig_vazia, fig_vazia, fig_vazia, fig_vazia
    
    fig_clusters = grafico_clusters(resultado['df'], resultado['centroids'])
    fig_importancia = grafico_importancia(resultado['importancia'])
    fig_cotovelo = grafico_cotovelo(resultado['df'], resultado['features'])
    fig_silhueta = grafico_silhueta(resultado['X_scaled'], resultado['clusters'], resultado['k'])
    fig_perfil = grafico_perfil_clusters(resultado['df'], resultado['features'])
    fig_radar = grafico_radar(resultado['df'], resultado['features'])
    
    return {'display': 'block'}, fig_clusters, fig_importancia, fig_cotovelo, fig_silhueta, fig_perfil, fig_radar