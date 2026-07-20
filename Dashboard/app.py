<<<<<<< HEAD
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

=======
from pages import (
    home, dataframes, filtros, agrupamentos, profiling,
    parquet, plots, kmeans, classificacao,
    eda, insights, advanced_classification, pipeline_carlos
)
>>>>>>> kassiane-silva
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from data_loader import data_manager
import webbrowser
from threading import Timer
<<<<<<< HEAD
from pages import (
    home, dataframes, filtros, agrupamentos, profiling,
    parquet, plots, kmeans, classificacao, 
    eda, insights, advanced_classification
)

=======

import sys
import os
import webbrowser
from threading import Timer

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

from data_loader import data_manager

from pages import (
    home,
    dataframes,
    filtros,
    agrupamentos,
    profiling,
    parquet,
    plots,
    kmeans,
    classificacao,
    eda,
    insights,
    advanced_classification
)


# ==========================================================
# CONFIGURAÇÃO DO APP
# ==========================================================

>>>>>>> kassiane-silva
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True
)

server = app.server


# ==========================================================
# CARREGAR DADOS
# ==========================================================

df = data_manager.load_data()

<<<<<<< HEAD
# CSS customizado
=======

# ==========================================================
# CSS CUSTOMIZADO
# ==========================================================

>>>>>>> kassiane-silva
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Dashboard Wearables</title>
        {%favicon%}
        {%css%}

        <style>
<<<<<<< HEAD
            body { background-color: #0D0D0D !important; color: #FFFFFF !important; }
            .card { background-color: #1A1A1A !important; border: 1px solid #2A2A2A !important; border-radius: 10px !important; }
=======

            body {
                background-color: #0D0D0D !important;
                color: #FFFFFF !important;
            }

            .card {
                background-color: #1A1A1A !important;
                border: 1px solid #2A2A2A !important;
                border-radius: 10px !important;
            }

>>>>>>> kassiane-silva
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

<<<<<<< HEAD
# ================== SIDEBAR ==================
sidebar = html.Div([
    html.H5("FitMatch", className="p-3 text-primary"),
    html.Hr(),
    dbc.Nav([
        dbc.NavLink("Home", href="/", active="exact"),
        dbc.NavLink("Dataframes", href="/dataframes", active="exact"),
        dbc.NavLink("EDA", href="/eda", active="exact"),
        dbc.NavLink("Profiling", href="/profiling", active="exact"),
        dbc.NavLink("Plots", href="/plots", active="exact"),
        dbc.NavLink("Parquet", href="/parquet", active="exact"),
        dbc.NavLink("Filtros", href="/filtros", active="exact"),
        dbc.NavLink("Agrupamentos", href="/agrupamentos", active="exact"),
        dbc.NavLink("K-Means", href="/kmeans", active="exact"),
        dbc.NavLink("Classificação", href="/classificacao", active="exact"),
        dbc.NavLink("Classificação Avançada", href="/advanced-classification", active="exact"),
        dbc.NavLink("Insights", href="/insights", active="exact"),
    ], vertical=True, pills=True),
], style={"position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "16rem", "padding": "2rem 1rem", "backgroundColor": "#1A1A1A"})

# ================== LAYOUT ==================
app.layout = dbc.Container([
    dcc.Location(id='url', refresh=False),
    dbc.Row([
        dbc.Col(sidebar, width=2),
        dbc.Col(html.Div(id='page-content', style={"padding": "2rem", "backgroundColor": "#0D0D0D", "minHeight": "100vh"}), width=10)
    ])
], fluid=True)

# ================== NAVEGAÇÃO ==================
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
=======

# ==========================================================
# SIDEBAR
# ==========================================================

sidebar = html.Div(

    [

        html.H5(
            "FitMatch",
            className="p-3 text-primary"
        ),

        html.Hr(),

        dbc.Nav(

            [

                dbc.NavLink(
                    "Home",
                    href="/",
                    active="exact"
                ),

                dbc.NavLink(
                    "Dataframes",
                    href="/dataframes",
                    active="exact"
                ),

                dbc.NavLink(
                    "EDA",
                    href="/eda",
                    active="exact"
                ),

                dbc.NavLink(
                    "Profiling",
                    href="/profiling",
                    active="exact"
                ),

                dbc.NavLink(
                    "Plots",
                    href="/plots",
                    active="exact"
                ),

                dbc.NavLink(
                    "Parquet",
                    href="/parquet",
                    active="exact"
                ),

                dbc.NavLink(
                    "Filtros",
                    href="/filtros",
                    active="exact"
                ),

                dbc.NavLink(
                    "Agrupamentos",
                    href="/agrupamentos",
                    active="exact"
                ),

                dbc.NavLink(
                    "K-Means",
                    href="/kmeans",
                    active="exact"
                ),

                dbc.NavLink(
                    "Classificação",
                    href="/classificacao",
                    active="exact"
                ),

                dbc.NavLink(
                    "Classificação Avançada",
                    href="/advanced-classification",
                    active="exact"
                ),

                dbc.NavLink(
                    "Insights",
                    href="/insights",
                    active="exact"
                ),

            ],

            vertical=True,
            pills=True

        )

    ],

    style={
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "16rem",
        "padding": "2rem 1rem",
        "backgroundColor": "#1A1A1A"
    }

)


# ==========================================================
# LAYOUT
# ==========================================================

app.layout = dbc.Container(

    [

        dcc.Location(
            id="url",
            refresh=False
        ),

        dbc.Row(

            [

                dbc.Col(
                    sidebar,
                    width=2
                ),


                dbc.Col(

                    html.Div(

                        id="page-content",

                        style={

                            "padding": "2rem",
                            "backgroundColor": "#0D0D0D",
                            "minHeight": "100vh"

                        }

                    ),

                    width=10

                )

            ]

        )

    ],

    fluid=True

)



# ==========================================================
# NAVEGAÇÃO
# ==========================================================

@app.callback(

    Output(
        "page-content",
        "children"
    ),

    Input(
        "url",
        "pathname"
    )

>>>>>>> kassiane-silva
)

def display_page(pathname):
<<<<<<< HEAD
=======

>>>>>>> kassiane-silva
    pages = {
        '/dataframes': dataframes.create_layout,
        '/filtros': filtros.create_layout,
        '/agrupamentos': agrupamentos.create_layout,
        '/profiling': profiling.create_layout,
        '/parquet': parquet.create_layout,
        '/plots': plots.create_layout,
<<<<<<< HEAD
=======
        '/kmeans': kmeans.create_layout,
>>>>>>> kassiane-silva
        '/classificacao': classificacao.create_layout,
        '/advanced-classification': advanced_classification.create_layout,
        '/kmeans': kmeans.create_layout,
        '/eda': eda.create_layout,
        '/insights': insights.create_layout,
        '/advanced-classification': advanced_classification.create_layout,
        '/pipeline-carlos': pipeline_carlos.create_layout,

        "/": home.create_layout,

        "/dataframes": dataframes.create_layout,

        "/filtros": filtros.create_layout,

        "/agrupamentos": agrupamentos.create_layout,

        "/profiling": profiling.create_layout,

        "/parquet": parquet.create_layout,

        "/plots": plots.create_layout,

        "/kmeans": kmeans.create_layout,

        "/classificacao": classificacao.create_layout,

        "/advanced-classification": advanced_classification.create_layout,

        "/eda": eda.create_layout,

        "/insights": insights.create_layout,

    }
<<<<<<< HEAD
    
    page_func = pages.get(pathname, home.create_layout)
    return page_func(df)

# ================== AUTO OPEN ==================
=======


    page_func = pages.get(
        pathname,
        home.create_layout
    )


    return page_func(df)



# ==========================================================
# ABRIR NAVEGADOR
# ==========================================================

>>>>>>> kassiane-silva
def open_browser():

    webbrowser.open_new(
        "http://localhost:8050"
    )

<<<<<<< HEAD
if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"✅ APLICATIVO INICIADO!")
    print(f"📎 Acesse: http://localhost:8050")
    print(f"{'='*60}\n")
=======


# ==========================================================
# EXECUÇÃO
# ==========================================================
>>>>>>> kassiane-silva

if __name__ == "__main__":


    print(
        "\n" + "=" * 60
    )

    print(
        "✅ APLICATIVO INICIADO!"
    )

    print(
        "📎 Acesse: http://localhost:8050"
    )

    print(
        "=" * 60 + "\n"
    )


    if os.environ.get(
        "WERKZEUG_RUN_MAIN"
    ) == "true":

        Timer(
            1,
            open_browser
        ).start()


    app.run(
        debug=True,
        host="0.0.0.0",
        port=8050
    )