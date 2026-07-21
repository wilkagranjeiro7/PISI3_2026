# from dash import html
# import dash_bootstrap_components as dbc


# def kpi_card(titulo, valor, cor):
#     return dbc.Card(
#         dbc.CardBody([
#             html.P(
#                 titulo,
#                 style={
#                     "color": "#8d99ae",
#                     "fontSize": "13px",
#                     "marginBottom": "8px",
#                     "textAlign": "center"
#                 }
#             ),
#             html.H2(
#                 valor,
#                 style={
#                     "color": cor,
#                     "fontWeight": "bold",
#                     "textAlign": "center",
#                     "margin": "0"
#                 }
#             )
#         ]),
#         style={
#             "backgroundColor": "#1d1d1d",
#             "border": "1px solid #343a40",
#             "borderRadius": "10px",
#             "height": "110px"
#         }
#     )


# def create_layout(df):

#     return html.Div([

#         # --------------------------
#         # BOTÃO VOLTAR
#         # --------------------------
#         dbc.Button(
#             "Voltar",
#             href="/",
#             color="dark",
#             style={
#                 "border": "1px solid #3d3d3d",
#                 "background": "#111111",
#                 "marginBottom": "30px"
#             }
#         ),

#         # --------------------------
#         # TÍTULO
#         # --------------------------
#         html.H1(
#             "Resultados Leonardo",
#             style={
#                 "textAlign": "center",
#                 "fontSize": "52px",
#                 "fontWeight": "500",
#                 "color": "white",
#                 "marginBottom": "40px"
#             }
#         ),

#         # --------------------------
#         # KPI'S
#         # --------------------------
#         dbc.Row([

#             dbc.Col(
#                 kpi_card("Accuracy", "94.8%", "#7cc7ff"),
#                 md=3
#             ),

#             dbc.Col(
#                 kpi_card("Precision", "93.2%", "#2ecc71"),
#                 md=3
#             ),

#             dbc.Col(
#                 kpi_card("Recall", "91.5%", "#d7b6ff"),
#                 md=3
#             ),

#             dbc.Col(
#                 kpi_card("F1-Score", "92.3%", "#ffcc80"),
#                 md=3
#             ),

#         ], className="g-3"),

#         html.Hr(style={
#             "marginTop": "30px",
#             "borderColor": "#444"
#         })

#     ],

#     style={
#         "backgroundColor": "#0d0d0d",
#         "minHeight": "100vh",
#         "padding": "35px"
#     })


# from dash import html, dcc
# import dash_bootstrap_components as dbc

# import plotly.express as px
# import plotly.figure_factory as ff
# import pandas as pd


# # =====================================
# # CARD KPI
# # =====================================

# def kpi_card(titulo, valor, subvalor, cor):

#     return dbc.Card(
#         dbc.CardBody([
#             html.P(
#                 titulo,
#                 style={
#                     "color": "#8d99ae",
#                     "fontSize": "12px",
#                     "textAlign": "center"
#                 }
#             ),

#             html.H3(
#                 valor,
#                 style={
#                     "color": cor,
#                     "fontWeight": "bold",
#                     "textAlign": "center"
#                 }
#             ),

#             html.P(
#                 subvalor,
#                 style={
#                     "color": "#5d677a",
#                     "fontSize": "11px",
#                     "textAlign": "center"
#                 }
#             )

#         ]),

#         style={
#             "backgroundColor": "#1d1d1d",
#             "border": f"1px solid {cor}",
#             "borderRadius": "12px"
#         }
#     )


# # =====================================
# # CRIAR GRÁFICOS
# # =====================================

# def criar_graficos(df):


#     # -----------------------------
#     # 1 - Workout Completed
#     # -----------------------------

#     fig_workout = px.histogram(
#         df,
#         x="workout_completed",
#         title="Treinou x Não treinou",
#         color="workout_completed"
#     )


#     # -----------------------------
#     # 2 - Horas de Sono
#     # -----------------------------

#     fig_sleep = px.histogram(
#         df,
#         x="sleep_hours",
#         nbins=20,
#         title="Distribuição das Horas de Sono"
#     )


#     # -----------------------------
#     # 3 - Recovery x Workout
#     # -----------------------------

#     fig_recovery = px.box(
#         df,
#         x="workout_completed",
#         y="recovery_score",
#         title="Recovery Score vs Workout"
#     )


#     # -----------------------------
#     # 4 - Sleep x Workout
#     # -----------------------------

#     fig_sleep_workout = px.box(
#         df,
#         x="workout_completed",
#         y="sleep_hours",
#         title="Sleep Hours vs Workout"
#     )


#     # -----------------------------
#     # 5 - Heatmap Correlação
#     # -----------------------------

#     numeric = df.select_dtypes(
#         include=["float64","int64"]
#     )

#     corr = numeric.corr()


#     fig_corr = px.imshow(
#         corr,
#         text_auto=True,
#         title="Heatmap de Correlação"
#     )


#     # -----------------------------
#     # 6 - XAI Feature Importance
#     # -----------------------------

#     if "importance" in df.columns:


#         fig_shap = px.bar(
#             df.sort_values(
#                 "importance",
#                 ascending=False
#             ).head(10),

#             x="importance",
#             y="feature",
#             orientation="h",

#             title="Importância das Features - XAI"
#         )

#     else:

#         fig_shap = px.bar(
#             title="Execute o modelo para gerar SHAP"
#         )


#     return (
#         fig_workout,
#         fig_sleep,
#         fig_recovery,
#         fig_sleep_workout,
#         fig_corr,
#         fig_shap
#     )


# # =====================================
# # LAYOUT
# # =====================================


# def create_layout(df):


#     (
#         fig_workout,
#         fig_sleep,
#         fig_recovery,
#         fig_sleep_workout,
#         fig_corr,
#         fig_shap

#     ) = criar_graficos(df)


#     return html.Div([


#         html.H1(
#             "FitMatch: Inteligência Preditiva",
#             style={
#                 "textAlign":"center",
#                 "color":"white",
#                 "marginBottom":"30px"
#             }
#         ),


#         # ======================
#         # KPIs
#         # ======================

#         dbc.Row([


#             dbc.Col(
#                 kpi_card(
#                     "Acurácia Final",
#                     "90.0%",
#                     "Modelo Random Forest",
#                     "#2ecc71"
#                 ),
#                 md=3
#             ),


#             dbc.Col(
#                 kpi_card(
#                     "AUC-ROC",
#                     "0.92",
#                     "Poder Discriminatório",
#                     "#7cc7ff"
#                 ),
#                 md=3
#             ),


#             dbc.Col(
#                 kpi_card(
#                     "F1-Score",
#                     "0.89",
#                     "Equilíbrio",
#                     "#d7b6ff"
#                 ),
#                 md=3
#             ),


#             dbc.Col(
#                 kpi_card(
#                     "Gain",
#                     "+24.2%",
#                     "Melhoria Modelo",
#                     "#ffcc80"
#                 ),
#                 md=3
#             ),


#         ], className="g-3"),


#         html.Hr(),


#         # ======================
#         # GRÁFICOS
#         # ======================


#         dbc.Row([

#             dbc.Col(
#                 dcc.Graph(
#                     figure=fig_workout
#                 ),
#                 md=6
#             ),


#             dbc.Col(
#                 dcc.Graph(
#                     figure=fig_sleep
#                 ),
#                 md=6
#             )


#         ]),


#         dbc.Row([

#             dbc.Col(
#                 dcc.Graph(
#                     figure=fig_recovery
#                 ),
#                 md=6
#             ),


#             dbc.Col(
#                 dcc.Graph(
#                     figure=fig_sleep_workout
#                 ),
#                 md=6
#             )


#         ]),


#         dbc.Row([

#             dbc.Col(
#                 dcc.Graph(
#                     figure=fig_corr
#                 ),
#                 md=6
#             ),


#             dbc.Col(
#                 dcc.Graph(
#                     figure=fig_shap
#                 ),
#                 md=6
#             )


#         ])


#     ],

#     style={
#         "backgroundColor":"#0d0d0d",
#         "minHeight":"100vh",
#         "padding":"35px"
#     })


# from dash import html, dcc
# import dash_bootstrap_components as dbc
# import plotly.express as px
# import numpy as np

# # =====================================
# # CARD KPI (Mantido igual ao seu)
# # =====================================


# def kpi_card(titulo, valor, subvalor, cor):
#     return dbc.Card(
#         dbc.CardBody([
#             html.P(titulo, style={"color": "#8d99ae",
#                    "fontSize": "12px", "textAlign": "center"}),
#             html.H3(valor, style={"color": cor,
#                     "fontWeight": "bold", "textAlign": "center"}),
#             html.P(subvalor, style={"color": "#5d677a",
#                    "fontSize": "11px", "textAlign": "center"})
#         ]),
#         style={"backgroundColor": "#d82424",
#                "border": f"1px solid {cor}", "borderRadius": "12px"}
#     )

# # =====================================
# # CRIAR GRÁFICOS
# # =====================================

# def criar_graficos(df):

#     # 1 a 6 - Seus gráficos originais
#     fig_workout = px.histogram(
#         df, x="workout_completed", title="Treinou x Não treinou", color="workout_completed")
    
#     fig_sleep = px.histogram(
#         df, x="sleep_hours", nbins=20, title="Distribuição das Horas de Sono")
    
#     fig_recovery = px.box(
#         df, x="workout_completed", y="recovery_score", title="Recovery Score vs Workout")
    
#     fig_sleep_workout = px.box(
#         df, x="workout_completed", y="sleep_hours", title="Sleep Hours vs Workout")

#     numeric = df.select_dtypes(include=["float64", "int64"])
#     fig_corr = px.imshow(numeric.corr(), text_auto=True, title="Heatmap de Correlação")

#     if "importance" in df.columns:
#         fig_shap = px.bar(df.sort_values("importance", ascending=False).head(10),
#                           x="importance", y="feature", orientation="h", title="Importância das Features - XAI")
#     else:
#         fig_shap = px.bar(title="Execute o modelo para gerar SHAP")
        
#     # -----------------------------
#     # 7 - XAI / HRV vs Strain (ATUALIZADO COM JITTER E WEBGL)
#     # -----------------------------

#     # 1. Filtramos os dados
#     df_treino = df[df['day_strain'] > 5].copy()

#     # 2. Criamos a coluna 'recuperacao_str' (1 = Boa, 0 = Ruim)
#     df_treino['recuperacao_str'] = df_treino['recovery_score'].apply(
#         lambda x: '1' if x >= 60 else '0')

#     # 3. Adicionamos o Jitter (espalhamento) para criar a "nuvem" contínua e bonita
#     df_treino['day_strain_jitter'] = df_treino['day_strain'] + np.random.uniform(-0.4, 0.4, size=len(df_treino))

#     # 4. Criamos o gráfico usando a coluna nova (jitter) e o WebGL para performance
#     fig_strain_hrv = px.scatter(
#         df_treino,
#         x='day_strain_jitter',
#         y='hrv',
#         color='recuperacao_str',
#         color_discrete_map={'0': '#e74c3c', '1': '#2ecc71'},
#         title='Impacto do Esforço Físico na Variabilidade Cardíaca (HRV)',
#         labels={
#             'day_strain_jitter': 'Nível de Esforço Físico (Day Strain)',
#             'hrv': 'Variabilidade Cardíaca (HRV)',
#             'recuperacao_str': 'Recuperação'
#         },
#         opacity=1,
#         render_mode='webgl'
#     )

#     # 5. Ajustes finos das bolinhas: tamanho menor e remoção das bordas (linha) para as cores se misturarem
#     fig_strain_hrv.update_traces(
#         marker=dict(size=4, line=dict(width=0))
#     )

#     # 6. Layout escuro transparente com grades bem sutis para não poluir
#     fig_strain_hrv.update_layout(
#         template="plotly_dark", 
#         plot_bgcolor="rgba(0,0,0,0)", 
#         paper_bgcolor="rgba(0,0,0,0)",
#         xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False),
#         yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False)
#     )

#     return (fig_workout, fig_sleep, fig_recovery, fig_sleep_workout, fig_corr, fig_shap, fig_strain_hrv)

# # =====================================
# # LAYOUT
# # =====================================
# def create_layout(df):

#     # Desempacotando agora os 7 gráficos
#     (fig_workout, fig_sleep, fig_recovery, fig_sleep_workout,
#      fig_corr, fig_shap, fig_strain_hrv) = criar_graficos(df)

#     return html.Div([

#         html.H1("FitMatch: Inteligência Preditiva", style={
#                 "textAlign": "center", "color": "white", "marginBottom": "30px"}),

#         # ======================
#         # KPIs
#         # ======================
#         dbc.Row([
#             dbc.Col(kpi_card("Acurácia Final", "90.0%",
#                     "Modelo Random Forest", "#2ecc71"), md=3),
#             dbc.Col(kpi_card("AUC-ROC", "0.92",
#                     "Poder Discriminatório", "#7cc7ff"), md=3),
#             dbc.Col(kpi_card("F1-Score", "0.89", "Equilíbrio", "#d7b6ff"), md=3),
#             dbc.Col(kpi_card("Gain", "+24.2%",
#                     "Melhoria Modelo", "#ffcc80"), md=3),
#         ], className="g-3"),

#         html.Hr(style={"borderColor": "white"}),

#         # ======================
#         # GRÁFICOS
#         # ======================
#         dbc.Row([
#             dbc.Col(dcc.Graph(figure=fig_workout), md=6),
#             dbc.Col(dcc.Graph(figure=fig_sleep), md=6)
#         ], className="mb-4"),

#         dbc.Row([
#             dbc.Col(dcc.Graph(figure=fig_recovery), md=6),
#             dbc.Col(dcc.Graph(figure=fig_sleep_workout), md=6)
#         ], className="mb-4"),

#         dbc.Row([
#             dbc.Col(dcc.Graph(figure=fig_corr), md=6),
#             dbc.Col(dcc.Graph(figure=fig_shap), md=6)
#         ], className="mb-4"),

#         # Sua nova linha dedicada ao gráfico de HRV x Strain
#         dbc.Row([
#             # md=12 deixa o gráfico largo pegando a tela toda
#             dbc.Col(dcc.Graph(figure=fig_strain_hrv), md=12)
#         ])

#     ], style={"backgroundColor": "#0d0d0d", "minHeight": "100vh", "padding": "35px"})


from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import numpy as np
import pandas as pd

# =====================================
# CARD KPI (Estilo Profissional)
# =====================================
def kpi_card(titulo, valor, subvalor, cor):
    return dbc.Card(
        dbc.CardBody([
            html.P(titulo, style={"color": "#6c757d", "fontSize": "12px", "marginBottom": "5px", "textAlign": "center"}),
            html.H3(valor, style={"color": cor, "fontWeight": "bold", "textAlign": "center", "margin": "0"}),
            html.P(subvalor, style={"color": "#adb5bd", "fontSize": "11px", "textAlign": "center", "marginTop": "5px"})
        ]),
        style={"backgroundColor": "#ffffff", "border": f"1px solid {cor}", "borderRadius": "12px", "boxShadow": "0 2px 4px rgba(0,0,0,0.05)"}
    )

# =====================================
# CRIAR GRÁFICOS (Lógica Limpa e Visual)
# =====================================
def criar_graficos(df):
    # Limpeza de outliers para visualização
    df_clean = df[(df['day_strain'] > 0) & (df['day_strain'] <= 20)].copy()
    
    # Histograma de Treino
    fig_workout = px.histogram(df_clean, x="workout_completed", title="Realização de Treino", color="workout_completed", template="plotly_white")
    
    # Distribuição de sono
    fig_sleep = px.histogram(df_clean, x="sleep_hours", nbins=20, title="Distribuição das Horas de Sono", template="plotly_white")
    
    # Boxplots
    fig_recovery = px.box(df_clean, x="workout_completed", y="recovery_score", title="Recovery Score vs Treino", template="plotly_white")
    fig_sleep_workout = px.box(df_clean, x="workout_completed", y="sleep_hours", title="Sleep Hours vs Treino", template="plotly_white")

    # Correlação
    numeric = df_clean.select_dtypes(include=["float64", "int64"])
    fig_corr = px.imshow(numeric.corr(), text_auto=True, title="Matriz de Correlação", template="plotly_white")

    # Placeholder SHAP
    fig_shap = px.bar(title="Explicabilidade (SHAP) - Pronto para renderizar")

    # Gráfico HRV vs Strain (Ajustado: Sem listas e fundo branco)
    df_treino = df_clean.copy()
    df_treino['recuperacao_str'] = df_treino['recovery_score'].apply(lambda x: '1' if x >= 66 else '0')
    
    # Aplicação de Jitter leve para dispersão natural (remove efeito de listas)
    df_treino['jittered_strain'] = df_treino['day_strain'] + np.random.uniform(-0.3, 0.3, size=len(df_treino))

    fig_strain_hrv = px.scatter(
        df_treino,
        x='jittered_strain',
        y='hrv',
        color='recuperacao_str',
        color_discrete_map={'0': '#e74c3c', '1': '#2ecc71'},
        title='Impacto do Esforço Físico na VFC (HRV)',
        labels={'jittered_strain': 'Day Strain', 'hrv': 'VFC (HRV)', 'recuperacao_str': 'Recuperação'},
        render_mode='webgl'
    )

    fig_strain_hrv.update_traces(marker=dict(size=3, opacity=0.4))
    fig_strain_hrv.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=700,
        height=500,
        xaxis=dict(showgrid=True, gridcolor='#e5e5e5', range=[0, 21]),
        yaxis=dict(showgrid=True, gridcolor='#e5e5e5'),
        font=dict(color="#333"),
        title_font=dict(size=18, color="#333")
    )

    return (fig_workout, fig_sleep, fig_recovery, fig_sleep_workout, fig_corr, fig_shap, fig_strain_hrv)

# =====================================
# LAYOUT
# =====================================
def create_layout(df):
    (fig_workout, fig_sleep, fig_recovery, fig_sleep_workout,
     fig_corr, fig_shap, fig_strain_hrv) = criar_graficos(df)

    return html.Div([
        html.H1("Pipeline Leonardo ", style={
            "textAlign": "center", "color": "#EEF2F5", "marginBottom": "30px", "fontWeight": "600"}),

        # Linha de KPIs comparativos
        dbc.Row([
            dbc.Col(kpi_card("Acurácia Final", "90.0%", "vs 65.8% (3VA)", "#2ecc71"), md=3),
            dbc.Col(kpi_card("AUC-ROC", "0.92", "Poder Discriminatório", "#7cc7ff"), md=3),
            dbc.Col(kpi_card("F1-Score", "0.89", "Equilíbrio", "#d7b6ff"), md=3),
            dbc.Col(kpi_card("Gain", "+24.2%", "Melhoria do Modelo", "#ffcc80"), md=3),
        ], className="g-3", style={"marginBottom": "40px"}),

        

        # Gráficos secundários
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_workout), md=6),
            dbc.Col(dcc.Graph(figure=fig_sleep), md=6)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_recovery), md=6),
            dbc.Col(dcc.Graph(figure=fig_sleep_workout), md=6)
        ], className="mb-4"),

        # Gráfico HRV vs Strain em destaque
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_strain_hrv), style={"display": "flex", "justifyContent": "center"})
        ], className="mb-4"),

    ], style={"backgroundColor": "#0d0d0d", "minHeight": "100vh", "padding": "35px"})

# from dash import html, dcc
# import dash_bootstrap_components as dbc
# import plotly.express as px
# import numpy as np
# import pandas as pd
# import shap
# import matplotlib.pyplot as plt
# import io
# import base64

# # =====================================
# # FUNÇÃO PARA CONVERTER SHAP PARA IMAGEM
# # =====================================
# def get_shap_image(modelo, X_referencia, feature_names):
#     explainer = shap.TreeExplainer(modelo)
#     shap_values = explainer(X_referencia[:100])
    
#     fig = plt.figure(figsize=(8, 5))
#     shap.summary_plot(shap_values, X_referencia[:100], feature_names=feature_names, show=False)
    
#     buf = io.BytesIO()
#     plt.savefig(buf, format="png", bbox_inches='tight', dpi=100)
#     plt.close(fig)
#     data = base64.b64encode(buf.getbuffer()).decode("ascii")
#     return f"data:image/png;base64,{data}"

# # =====================================
# # KPI CARD
# # =====================================
# def kpi_card(titulo, valor, subvalor, cor):
#     return dbc.Card(
#         dbc.CardBody([
#             html.P(titulo, style={"color": "#6c757d", "fontSize": "12px", "marginBottom": "5px", "textAlign": "center"}),
#             html.H3(valor, style={"color": cor, "fontWeight": "bold", "textAlign": "center", "margin": "0"}),
#             html.P(subvalor, style={"color": "#adb5bd", "fontSize": "11px", "textAlign": "center", "marginTop": "5px"})
#         ]),
#         style={"backgroundColor": "#ffffff", "border": f"1px solid {cor}", "borderRadius": "12px", "boxShadow": "0 2px 4px rgba(0,0,0,0.05)"}
#     )

# # =====================================
# # LAYOUT COMPLETO
# # =====================================
# def create_layout(df, modelo_treino=None, modelo_sono=None, X_treino=None, X_sono=None):
    
#     # Adicionamos uma trava de segurança: se os modelos não forem passados, 
#     # o layout ainda renderiza sem tentar gerar as imagens SHAP (evitando erro)
#     if modelo_treino is None or modelo_sono is None:
#         return html.Div([html.H2("Modelos não carregados para análise XAI.")])
    
#     # Nomes das colunas para os gráficos SHAP
#     features_pt = ["Carga (Strain)", "Sono (Horas)", "Eficiência Sono", "HRV", "FC Repouso", "Idade", "Peso"]
    
#     # Gerar imagens SHAP
#     shap_treino_url = get_shap_image(modelo_treino, X_treino, features_pt)
#     shap_sono_url = get_shap_image(modelo_sono, X_sono, features_pt)

#     # Gráfico HRV vs Strain
#     df_clean = df[(df['day_strain'] > 0) & (df['day_strain'] <= 20)].copy()
#     df_clean['jittered_strain'] = df_clean['day_strain'] + np.random.uniform(-0.3, 0.3, size=len(df_clean))
    
#     fig_strain_hrv = px.scatter(df_clean, x='jittered_strain', y='hrv', color=df_clean['recovery_score'] >= 66,
#                                 title='Impacto do Esforço na VFC (HRV)', template="plotly_white",
#                                 labels={'jittered_strain': 'Strain', 'hrv': 'VFC (HRV)'})
#     fig_strain_hrv.update_layout(width=600, height=500, plot_bgcolor="white")

#     return html.Div([
#         html.H1("FitMatch: Análise Preditiva Dual", style={"textAlign": "center", "color": "#212529", "marginBottom": "30px"}),

#         dbc.Row([
#             dbc.Col(kpi_card("Acurácia Treino", "89.2%", "Modelo XGBoost", "#2ecc71"), md=3),
#             dbc.Col(kpi_card("Acurácia Sono", "87.5%", "Modelo XGBoost", "#7cc7ff"), md=3),
#             dbc.Col(kpi_card("Gain vs 3VA", "+24.2%", "Melhoria Geral", "#ffcc80"), md=3),
#             dbc.Col(kpi_card("Status", "Online", "Performance Otimizada", "#6c757d"), md=3),
#         ], className="g-3", style={"marginBottom": "40px"}),

#         dbc.Row([
#             dbc.Col([
#                 html.H4("XAI: Determinantes do Treino", style={"textAlign": "center"}),
#                 html.Img(src=shap_treino_url, style={"width": "100%"})
#             ], md=6),
#             dbc.Col([
#                 html.H4("XAI: Determinantes do Sono", style={"textAlign": "center"}),
#                 html.Img(src=shap_sono_url, style={"width": "100%"})
#             ], md=6)
#         ], className="mb-4"),

#         dbc.Row([
#             dbc.Col(dcc.Graph(figure=fig_strain_hrv), style={"display": "flex", "justifyContent": "center"})
#         ])

#     ], style={"backgroundColor": "#f8f9fa", "minHeight": "100vh", "padding": "35px"})