from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from data_loader import data_manager
import webbrowser
from threading import Timer

from pages import (
    home, dataframes, filtros, agrupamentos, boolean, profiling,
    parquet, plots, subplots, kmeans, classificacao, 
    matriz_confusao, eda
)

# Inicializar app
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
server = app.server

# 🔥 Função para obter DataFrame atual
def get_current_df():
    """Retorna o DataFrame atual do data_manager"""
    if hasattr(data_manager, 'get_current_df'):
        return data_manager.get_current_df()
    return data_manager.df

# Registrar callbacks (apenas uma vez)
matriz_confusao.register_callbacks(app)
kmeans.register_kmeans_callbacks(app, get_current_df)

# Carregar dados
df = data_manager.load_data()

# Layout principal com sidebar
app.layout = dbc.Container([
    dbc.Row([
        # Sidebar
        dbc.Col([
            html.H4("Menu Principal", className="text-center mt-3"),
            html.Hr(),
            dbc.Nav([
                dbc.NavLink("🏠 Home", href="/", active="exact"),
                dbc.NavLink("📊 DataFrames", href="/dataframes", active="exact"),
                dbc.NavLink("🔍 Filtros", href="/filtros", active="exact"),
                dbc.NavLink("📈 Agrupamentos", href="/agrupamentos", active="exact"),
                dbc.NavLink("✅ Boolean", href="/boolean", active="exact"),
                dbc.NavLink("🔬 Profiling", href="/profiling", active="exact"),
                dbc.NavLink("💾 Parquet", href="/parquet", active="exact"),
                dbc.NavLink("📉 Plots", href="/plots", active="exact"),
                dbc.NavLink("📊 Subplots", href="/subplots", active="exact"),
                dbc.NavLink("🎯 K-means", href="/kmeans", active="exact"),
                dbc.NavLink("🤖 Classificação", href="/classificacao", active="exact"),
                dbc.NavLink("📐 Matriz Confusão", href="/matriz-confusao", active="exact"),
                dbc.NavLink("📚 EDA", href="/eda", active="exact"),
            ], vertical=True, pills=True),
        ], width=2, style={
            'backgroundColor': '#f8f9fa',
            'minHeight': '100vh',
            'padding': '20px'
        }),
        
        # Conteúdo principal
        dbc.Col([
            dcc.Location(id='url', refresh=False),
            html.Div(id='page-content', style={'padding': '20px'})
        ], width=10)
    ])
], fluid=True)

# Callback para navegação
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    current_df = get_current_df()
    
    if current_df is None or current_df.empty:
        return html.Div([
            html.H1("⚠️ Nenhum dado carregado", className="text-danger"),
            html.P("Por favor, carregue um dataset na página inicial.")
        ])
    
    if pathname == '/dataframes':
        return dataframes.create_layout(current_df)
    elif pathname == '/filtros':
        return filtros.create_layout(current_df)
    elif pathname == '/agrupamentos':
        return agrupamentos.create_layout(current_df)
    elif pathname == '/boolean':
        return boolean.create_layout(current_df)
    elif pathname == '/profiling':
        return profiling.create_layout(current_df)
    elif pathname == '/parquet':
        return parquet.create_layout(current_df)
    elif pathname == '/plots':
        return plots.create_layout(current_df)
    elif pathname == '/subplots':
        return subplots.create_layout(current_df)
    elif pathname == '/kmeans':
        return kmeans.create_layout(current_df)
    elif pathname == '/classificacao':
        return classificacao.create_layout(current_df)
    elif pathname == '/matriz-confusao':
        return matriz_confusao.create_layout(current_df)
    elif pathname == '/eda':
        return eda.create_layout(current_df)
    else:
        return home.create_layout(current_df)

def open_browser():
    webbrowser.open_new("http://localhost:8050")

if __name__ == '__main__':
    import os

    print(f"\n{'='*60}")
    print(f"✅ APLICATIVO INICIADO!")
    print(f"📍 Acesse: http://localhost:8050")
    print(f"📊 DataFrame inicial: {df.shape if df is not None else 'None'}")
    print(f"{'='*60}\n")

    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        Timer(1, open_browser).start()

    app.run(debug=True, host='0.0.0.0', port=8050)