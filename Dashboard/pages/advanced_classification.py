import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.figure_factory as ff
import joblib
import pandas as pd

# Removemos o dash.register_page porque seu app usa roteamento manual!

def create_layout(df):
    # Carregar os resultados gerados pelo pipeline
    try:
        results = joblib.load('classification_results.pkl')
        metrics = results['metrics']
        dist_antes = results['dist_antes']
        dist_depois = results['dist_depois']
        shap_data = pd.DataFrame(results['shap_summary'])
    except FileNotFoundError:
        return html.Div("Resultados não encontrados. Rode o train_pipeline.py primeiro.")

    # Gráfico 1: Balanceamento
    df_antes = pd.DataFrame(list(dist_antes.items()), columns=['Classe', 'Quantidade'])
    df_antes['Status'] = 'Antes do SMOTE'
    df_depois = pd.DataFrame(list(dist_depois.items()), columns=['Classe', 'Quantidade'])
    df_depois['Status'] = 'Depois do SMOTE'
    df_bal = pd.concat([df_antes, df_depois])
    fig_balance = px.bar(df_bal, x='Classe', y='Quantidade', color='Status', barmode='group', title="Distribuição do Target")

    # Gráfico 2: Matriz de Confusão
    z = metrics['conf_matrix']
    x = metrics['classes']
    y = metrics['classes']
    fig_cm = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues')
    fig_cm.update_layout(title_text='Matriz de Confusão', xaxis_title="Predito", yaxis_title="Real")

    # Gráfico 3: SHAP
    fig_shap = px.bar(shap_data, x='Importance', y='Feature', orientation='h', title="Impacto das Variáveis (SHAP)")
    fig_shap.update_layout(yaxis={'categoryorder':'total ascending'})

    return dbc.Container([
        html.H2("Classificação Avançada com Pipeline e XAI", className="mt-4 mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody([html.H5("Novo Modelo (Random Forest)"), html.P(f"Acurácia: {metrics['accuracy']:.1%}"), html.P(f"F1-Score: {metrics['f1_score']:.3f}")])], color="success", outline=True), width=6),
            dbc.Col(dbc.Card([dbc.CardBody([html.H5("Melhor Modelo Anterior (Gradient Boosting 3VA)"), html.P("Acurácia: 58.2%"), html.P("F1-Score: 0.585")])], color="secondary", outline=True), width=6),
        ], className="mb-4"),
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_balance), width=6), dbc.Col(dcc.Graph(figure=fig_cm), width=6)]),
        html.H4("Explicabilidade (XAI)"),
        dcc.Graph(figure=fig_shap)
    ], fluid=True)