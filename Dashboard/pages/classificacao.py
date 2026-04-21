# pages/classificacao.py
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def create_layout(df):
    """Página para modelos de classificação"""
    
    # Preparar dados para classificação
    df_ml = df.dropna().copy()
    df_ml['target'] = (df_ml['recovery_score'] > 66).astype(int)  # 1 = Alto, 0 = Baixo
    
    features = ['day_strain', 'sleep_hours', 'sleep_efficiency', 'hrv', 'resting_heart_rate']
    
    return html.Div([
        html.H1("🤖 Classificação de Recovery Score", className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Configurações do Modelo"),
                    dbc.CardBody([
                        html.Label("Features selecionadas:"),
                        dcc.Dropdown(
                            id='class-features',
                            options=[{'label': f, 'value': f} for f in features],
                            value=features,
                            multi=True
                        ),
                        html.Br(),
                        
                        html.Label("Tamanho do teste (%):"),
                        dcc.Slider(id='class-test-size', min=0.1, max=0.4, step=0.05,
                                  value=0.2, marks={0.1: '10%', 0.2: '20%', 0.3: '30%', 0.4: '40%'}),
                        html.Br(),
                        
                        html.Label("Número de árvores:"),
                        dcc.Slider(id='class-n-estimators', min=50, max=200, step=50,
                                  value=100, marks={50: '50', 100: '100', 150: '150', 200: '200'})
                    ])
                ])
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Resultados da Classificação"),
                    dbc.CardBody([
                        html.Div(id='class-results'),
                        dcc.Graph(id='class-feature-importance')
                    ])
                ])
            ], width=9)
        ])
    ])

@callback(
    Output('class-results', 'children'),
    Output('class-feature-importance', 'figure'),
    [Input('class-features', 'value'),
     Input('class-test-size', 'value'),
     Input('class-n-estimators', 'value')]
)
def update_classification(features, test_size, n_estimators):
    from data_loader import data_manager
    df = data_manager.df.dropna().copy()
    
    if len(features) < 2:
        return html.P("Selecione pelo menos 2 features"), go.Figure()
    
    # Preparar dados
    df['target'] = (df['recovery_score'] > 66).astype(int)
    X = df[features]
    y = df['target']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Treinar modelo
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X_train, y_train)
    
    # Avaliar
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Resultados
    results = html.Div([
        html.H5(f"🎯 Acurácia: {accuracy:.2%}"),
        html.P(f"📊 Amostras de treino: {len(X_train):,}"),
        html.P(f"📊 Amostras de teste: {len(X_test):,}"),
        html.Hr(),
        html.H6("Relatório de Classificação:"),
        html.Pre(classification_report(y_test, y_pred, 
                                      target_names=['Baixo Recovery', 'Alto Recovery']))
    ])
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=True)
    
    fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                 title='Importância das Features',
                 color='importance', color_continuous_scale='Viridis')
    fig.update_layout(template='plotly_white')
    
    return results, fig