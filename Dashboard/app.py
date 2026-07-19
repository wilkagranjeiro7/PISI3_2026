from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from data_loader import data_manager
import webbrowser
from threading import Timer
from pages import (
    home, dataframes, filtros, agrupamentos, profiling,
    parquet, plots, kmeans, classificacao,
    eda, insights, pipeline_carlos
)


app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True
)
server = app.server

# Carregar dados
df = data_manager.load_data()

# CSS customizado para dropdowns e scrollbar
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Dashboard Wearables</title>
        {%favicon%}
        {%css%}
        <style>
            /* Reset e cores base */
            body {
                background-color: #0D0D0D !important;
                color: #FFFFFF !important;
            }
            
            /* Estilo para dropdowns */
            .Select-control {
                background-color: #1A1A1A !important;
                border: 1px solid #2A2A2A !important;
                border-radius: 4px !important;
            }
            
            /* Garantir que o texto selecionado fique branco */
            .Select-value-label,
            .has-value.Select--single > .Select-control .Select-value .Select-value-label,
            .Select--single > .Select-control .Select-value .Select-value-label {
                color: #FFFFFF !important;
            }
            
            .Select-control .Select-placeholder {
                color: #888888 !important;
            }
            
            .Select-menu-outer {
                background-color: #1A1A1A !important;
                border: 1px solid #2A2A2A !important;
                border-radius: 4px !important;
            }
            
            .Select-option {
                background-color: #1A1A1A !important;
                color: #888888 !important;
            }
            
            .Select-option.is-focused {
                background-color: #2A2A2A !important;
                color: #FFFFFF !important;
            }
            
            .Select-option.is-selected {
                background-color: #3B82F6 !important;
                color: #FFFFFF !important;
            }
            
            /* Fundo do controle */
            .Select,
            .Select-multi-value-wrapper {
                background-color: #1A1A1A !important;
            }
            
            .Select-value {
                background-color: #1A1A1A !important;
                color: #FFFFFF !important;
            }
            
            /* Para os novos dropdowns do Dash */
            .dash-dropdown .Select-value-label {
                color: #FFFFFF !important;
            }
            
            .VirtualizedSelectOption {
                background-color: #1A1A1A !important;
                color: #888888 !important;
            }
            
            .VirtualizedSelectFocusedOption {
                background-color: #2A2A2A !important;
                color: #FFFFFF !important;
            }
            
            /* Estilo para scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            
            ::-webkit-scrollbar-track {
                background: #1A1A1A;
                border-radius: 4px;
            }
            
            ::-webkit-scrollbar-thumb {
                background: #2A2A2A;
                border-radius: 4px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: #3B82F6;
            }
            
            /* Estilo para sliders */
            .rc-slider-track {
                background-color: #3B82F6 !important;
            }
            
            .rc-slider-handle {
                border-color: #3B82F6 !important;
                background-color: #3B82F6 !important;
            }
            
            .rc-slider-rail {
                background-color: #2A2A2A !important;
            }
            
            .rc-slider-mark-text {
                color: #888888 !important;
            }
            
            /* Estilo para abas e tabs */
            .tab {
                background-color: #1A1A1A !important;
                border: 1px solid #2A2A2A !important;
                color: #888888 !important;
            }
            
            .tab--selected {
                background-color: #0D0D0D !important;
                color: #FFFFFF !important;
                border-bottom: 2px solid #3B82F6 !important;
            }
            
            /* Estilo para cards e containers */
            .card {
                background-color: #1A1A1A !important;
                border: 1px solid #2A2A2A !important;
                border-radius: 10px !important;
            }
            
            .card-title {
                color: #FFFFFF !important;
            }
            
            .card-text {
                color: #888888 !important;
            }
            
            /* Links e botões */
            a {
                color: #3B82F6 !important;
                text-decoration: none !important;
            }
            
            a:hover {
                color: #60A5FA !important;
            }
            
            /* Inputs e textarea */
            input, textarea {
                background-color: #1A1A1A !important;
                border: 1px solid #2A2A2A !important;
                color: #FFFFFF !important;
            }
            
            /* Tabelas */
            table {
                background-color: #1A1A1A !important;
                border-color: #2A2A2A !important;
                color: #FFFFFF !important;
            }
            
            th, td {
                border-color: #2A2A2A !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ================== LAYOUT ==================
app.layout = dbc.Container([

    dcc.Location(id='url', refresh=False),

    html.Div(
        id='page-content',
        style={
            'padding': '20px',
            'backgroundColor': '#0D0D0D',
            'minHeight': '100vh',
            'color': '#FFFFFF'
        }
    )

], fluid=True)


# ================== NAVEGAÇÃO ==================
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    # Dicionário de rotas
    pages = {
        '/dataframes': dataframes.create_layout,
        '/filtros': filtros.create_layout,
        '/agrupamentos': agrupamentos.create_layout,
        '/profiling': profiling.create_layout,
        '/parquet': parquet.create_layout,
        '/plots': plots.create_layout,
        '/kmeans': kmeans.create_layout,
        '/classificacao': classificacao.create_layout,
        '/eda': eda.create_layout,
        '/insights': insights.create_layout,
        '/pipeline-carlos': pipeline_carlos.create_layout,
    }
    
    # Página padrão (home)
    page_func = pages.get(pathname, home.create_layout)
    
    # Retorna a página com o DataFrame
    return page_func(df)


# ================== AUTO OPEN ==================
def open_browser():
    webbrowser.open_new("http://localhost:8050")


if __name__ == '__main__':
    import os

    print(f"\n{'='*60}")
    print(f"✅ APLICATIVO INICIADO!")
    print(f"📎 Acesse: http://localhost:8050")
    print(f"{'='*60}\n")

    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        Timer(1, open_browser).start()

    app.run(debug=True, host='0.0.0.0', port=8050)