

# from dash import html
# import dash_bootstrap_components as dbc


# def create_layout(df):

#     return html.Div([

#         html.H1(
#             "Pipeline Leonardo",
#             style={
#                 "textAlign": "center",
#                 "color": "#EEF2F5",
#                 "marginBottom": "30px"
#             }
#         ),


#         dbc.Row([

#             dbc.Col([
#                 html.H4(
#                     "Gráfico 1",
#                     style={"color": "white", "textAlign": "center"}
#                 ),
#                 html.Img(
#                     src="/assets/Distribuição-treino-realizado.PNG",
#                     style={
#                         "width": "400px",
#                         'heigh': "400px",
#                     }
#                 )
#             ], md=6),

#             dbc.Col([
#                 html.H4(
#                     "Gráfico 2",
#                     style={"color": "white", "textAlign": "center"}
#                 ),
#                 html.Img(
#                     src="/assets/target2.PNG",
#                     style={
#                         "width": "400px",
#                         'heigh': "400px",
#                         "borderRadius": "12px"
#                     }
#                 )
#             ], md=6),

#         ], className="mb-4"),


#         dbc.Row([

#             dbc.Col([
#                 html.H4(
#                     "Gráfico 3",
#                     style={"color": "white", "textAlign": "center"}
#                 ),
#                 html.Img(
#                     src="/assets/curva_roc_comparativa.PNG",
#                     style={
#                         "width": "100%",
#                         "borderRadius": "12px"
#                     }
#                 )
#             ], md=6),

#             dbc.Col([
#                 html.H4(
#                     "Gráfico 4",
#                     style={"color": "white", "textAlign": "center"}
#                 ),
#                 html.Img(
#                     src="assets/Smote.PNG",
#                     style={
#                         "width": "100%",

#                     }
#                 )
#             ], md=6),

#         ], className="mb-4"),

#         dbc.Row([

#             dbc.Col([
#                 html.H4(
#                     "Gráfico 3",
#                     style={"color": "white", "textAlign": "center"}
#                 ),
#                 html.Img(
#                     src="/assets/confusao_modelos_performace.PNG",
#                     style={
#                         "width": "100%",
#                         "borderRadius": "12px"
#                     }
#                 )
#             ], md=12),



#         ], className="mb-4"),


#         dbc.Row([

#             dbc.Col([
#                 html.H4(
#                     "Gráfico 1",
#                     style={"color": "white", "textAlign": "center"}
#                 ),
#                 html.Img(
#                     src="/assets/fatores_determinantes.PNG",
#                     style={
#                         "width": "400px",
#                         'heigh': "400px",
#                     }
#                 )
#             ], md=6),

#             dbc.Col([
#                 html.H4(
#                     "Gráfico 2",
#                     style={"color": "white", "textAlign": "center"}
#                 ),
#                 html.Img(
#                     src="/assets/SHAP.PNG",
#                     style={
#                         "width": "400px",
#                         'heigh': "400px",
#                         "borderRadius": "12px"
#                     }
#                 )
#             ], md=6),

#         ], className="mb-4"),



#     ],
#         style={
#         "backgroundColor": "#0d0d0d",
#         "minHeight": "100vh",
#         "padding": "35px"
#     })


from dash import html
import dash_bootstrap_components as dbc

def create_layout(df):
    # Estilos padronizados para imagens
    IMAGE_STYLE = {
        "width": "100%",
        "height": "350px",  # Altura fixa para todas as imagens
        "objectFit": "contain",  # Mantém a proporção sem cortar
        "borderRadius": "12px",
        "backgroundColor": "#1a1a1a",
        "padding": "10px",
        "border": "1px solid #2a2a2a",
        "transition": "all 0.3s ease"
    }
    
    # Estilo para os containers das imagens
    CARD_STYLE = {
        "backgroundColor": "#1a1a1a",
        "borderRadius": "16px",
        "padding": "20px",
        "border": "1px solid #2a2a2a",
        "boxShadow": "0 4px 20px rgba(0,0,0,0.3)",
        "height": "100%",
        "transition": "all 0.3s ease"
    }
    
    # Estilo para títulos
    TITLE_STYLE = {
        "color": "#EEF2F5",
        "textAlign": "center",
        "marginBottom": "15px",
        "fontSize": "18px",
        "fontWeight": "600",
        "letterSpacing": "0.5px"
    }

    return html.Div([
        # Cabeçalho com ícone
        html.Div([
            html.I(className="fas fa-robot", style={
                "color": "#4A90E2",
                "fontSize": "40px",
                "marginRight": "15px"
            }),
            html.H1(
                "Pipeline Leonardo - Visualização de Resultados",
                style={
                    "textAlign": "center",
                    "color": "#EEF2F5",
                    "marginBottom": "10px",
                    "fontWeight": "700",
                    "letterSpacing": "1px"
                }
            ),
        ], style={"display": "flex", "alignItems": "center", "justifyContent": "center", "marginBottom": "30px"}),
        
        html.P(
            "Análise de dados e visualização dos resultados do pipeline",
            style={
                "textAlign": "center",
                "color": "#8899AA",
                "marginBottom": "40px",
                "fontSize": "16px"
            }
        ),

        # Primeira linha - Gráficos 1 e 2
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4("Distribuição de Treino", style=TITLE_STYLE),
                    html.Img(
                        src="/assets/Distribuição-treino-realizado.PNG",
                        style=IMAGE_STYLE
                    ),
                ], style=CARD_STYLE)
            ], md=6, className="mb-4"),
            
            dbc.Col([
                html.Div([
                    html.H4(" Análise de Target", style=TITLE_STYLE),
                    html.Img(
                        src="/assets/target2.PNG",
                        style=IMAGE_STYLE
                    ),
                ], style=CARD_STYLE)
            ], md=6, className="mb-4"),
        ], className="mb-3"),

        # Segunda linha - Gráficos 3 e 4
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4(" Curva ROC Comparativa", style=TITLE_STYLE),
                    html.Img(
                        src="/assets/curva_roc_comparativa.PNG",
                        style=IMAGE_STYLE
                    ),
                ], style=CARD_STYLE)
            ], md=6, className="mb-4"),
            
            dbc.Col([
                html.Div([
                    html.H4(" SMOTE Balanceamento", style=TITLE_STYLE),
                    html.Img(
                        src="/assets/Smote.PNG",
                        style=IMAGE_STYLE
                    ),
                ], style=CARD_STYLE)
            ], md=6, className="mb-4"),
        ], className="mb-3"),

        # Terceira linha - Gráfico 5 (largura total)
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4(" Matriz de Confusão - Performance dos Modelos", style={
                        **TITLE_STYLE,
                        "fontSize": "20px"
                    }),
                    html.Img(
                        src="/assets/confusao_modelos_performace.PNG",
                        style={
                            **IMAGE_STYLE,
                            "height": "420px"  # Um pouco maior por ser único
                        }
                    ),
                ], style=CARD_STYLE)
            ], md=12, className="mb-4"),
        ], className="mb-3"),

        # Quarta linha - Gráficos 6 e 7
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4(" Fatores Determinantes", style=TITLE_STYLE),
                    html.Img(
                        src="/assets/fatores_determinantes.PNG",
                        style=IMAGE_STYLE
                    ),
                ], style=CARD_STYLE)
            ], md=6, className="mb-4"),
            
            dbc.Col([
                html.Div([
                    html.H4("Análise SHAP", style=TITLE_STYLE),
                    html.Img(
                        src="/assets/SHAP.PNG",
                        style=IMAGE_STYLE
                    ),
                ], style=CARD_STYLE)
            ], md=6, className="mb-4"),
        ], className="mb-3"),

        # Rodapé com informações
        html.Div([
            html.Hr(style={"borderColor": "#2a2a2a", "marginTop": "30px"}),
            html.Div([
                html.Span(" Última atualização: "),
                html.Span("Hoje", style={"color": "#4A90E2", "fontWeight": "600"}),
                html.Span("  •  ", style={"color": "#2a2a2a"}),
                html.Span(" Total de gráficos: 7"),
                html.Span("  •  ", style={"color": "#2a2a2a"}),
                html.Span(" Pipeline concluído com sucesso", style={"color": "#27AE60"})
            ], style={
                "textAlign": "center",
                "color": "#8899AA",
                "fontSize": "14px",
                "padding": "20px 0"
            })
        ])

    ], style={
        "backgroundColor": "#0d0d0d",
        "minHeight": "100vh",
        "padding": "35px",
        "backgroundImage": "radial-gradient(circle at 10% 20%, rgba(74, 144, 226, 0.05) 0%, transparent 50%)"
    })