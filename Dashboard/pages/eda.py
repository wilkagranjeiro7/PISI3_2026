# pages/eda.py
from dash import html, dcc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
from data_loader import data_manager


# ==================================================
# CORES PADRÃO (via DataManager)
# ==================================================

CORES = data_manager.get_cores()
TEMPLATE = 'plotly_dark'


# ==================================================
# PREPROCESSAMENTO
# ==================================================

def preprocess_data(df):
    """Pré-processamento manual que funciona"""
    
    # Usar dados já limpos do DataManager se disponível
    df_clean = data_manager.get_clean_df()
    if df_clean is not None:
        df = df_clean.copy()
    else:
        df = df.copy()

    # ==================================
    # TREINOU
    # ==================================
    if "workout_time_of_day" in df.columns:
        df["treinou"] = df["workout_time_of_day"].notna().astype(int)
    else:
        df["treinou"] = 0

    # ==================================
    # IDADE
    # ==================================
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df.loc[(df["age"] < 10) | (df["age"] > 100), "age"] = pd.NA

    # ==================================
    # RECOVERY
    # ==================================
    if "recovery_score" in df.columns:
        df["recovery_score"] = pd.to_numeric(df["recovery_score"], errors="coerce")
        df.loc[(df["recovery_score"] < 0) | (df["recovery_score"] > 100), "recovery_score"] = pd.NA

    # ==================================
    # SONO
    # ==================================
    if "sleep_hours" in df.columns:
        df["sleep_hours"] = pd.to_numeric(df["sleep_hours"], errors="coerce")
        df.loc[(df["sleep_hours"] < 2) | (df["sleep_hours"] > 16), "sleep_hours"] = pd.NA

    # ==================================
    # FITNESS LEVEL
    # ==================================
    if "fitness_level" in df.columns:
        df["fitness_level"] = df["fitness_level"].apply(
            lambda x: data_manager.VALUE_TRANSLATIONS.get(str(x).lower(), x) if pd.notna(x) else x
        )

    # ==================================
    # HORÁRIO DO TREINO
    # ==================================
    if "workout_time_of_day" in df.columns:
        df["workout_time_of_day"] = df["workout_time_of_day"].apply(
            lambda x: data_manager.VALUE_TRANSLATIONS.get(str(x).lower(), x) if pd.notna(x) else x
        )

    # ==================================
    # TRADUÇÕES
    # ==================================
    if "gender" in df.columns:
        df["gender"] = df["gender"].apply(
            lambda x: data_manager.VALUE_TRANSLATIONS.get(str(x).lower(), x) if pd.notna(x) else x
        )

    # Traduzir nomes das colunas
    df.columns = [data_manager.traduzir_coluna(c) for c in df.columns]

    return df


# ==================================================
# FAIXA ETÁRIA
# ==================================================

def add_age_group(df):
    df = df.copy()
    
    if "Idade" in df.columns:
        age_col = "Idade"
    elif "age" in df.columns:
        age_col = "age"
    else:
        return df

    bins = [0, 20, 30, 40, 50, 60, 100]
    labels = ["Menos de 20", "20-29", "30-39", "40-49", "50-59", "60+"]
    df["faixa_idade"] = pd.cut(df[age_col], bins=bins, labels=labels)
    return df


# ==================================================
# LOLLIPOP CHART
# ==================================================

def create_lollipop(df, x_col, y_col, color_col, title, x_axis_title, y_axis_title):
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return html.Div("Sem dados para este gráfico", style={'color': CORES['text_secondary'], 'textAlign': 'center', 'padding': '20px'})
    
    fig = go.Figure()

    for cat in df[color_col].dropna().unique():
        subset = df[df[color_col] == cat]
        
        if cat == 'Masculino':
            cor = CORES['masculino']
        elif cat == 'Feminino':
            cor = CORES['feminino']
        else:
            cor = CORES['chart_colors'][0]

        fig.add_trace(go.Scatter(
            x=subset[x_col], y=subset[y_col],
            mode='markers+lines', name=cat,
            marker=dict(size=12, color=cor, line=dict(width=1, color=CORES['card_bg'])),
            line=dict(width=2, color=cor)
        ))

    fig.update_layout(
        title=title, template=TEMPLATE,
        xaxis_title=x_axis_title, yaxis_title=y_axis_title,
        paper_bgcolor=CORES['background'], plot_bgcolor=CORES['background'],
        font_color=CORES['text'], title_font_color=CORES['text'],
        hovermode='closest',
        xaxis=dict(gridcolor=CORES['border'], showgrid=True),
        yaxis=dict(gridcolor=CORES['border'], showgrid=True)
    )
    return dcc.Graph(figure=fig)


# ==================================================
# KPI CARDS
# ==================================================

def kpi_card(title, value, color=CORES['chart_colors'][0]):
    return html.Div([
        html.H5(title, style={'color': CORES['text_secondary'], 'marginBottom': '10px', 'fontSize': '14px'}),
        html.H2(value, style={'color': color, 'marginBottom': '0', 'fontWeight': 'bold'})
    ], style={
        "background": CORES['card_bg'], "color": CORES['text'],
        "border": f"1px solid {CORES['border']}",
        "padding": "20px", "borderRadius": "10px", "textAlign": "center",
        "flex": "1", "minWidth": "220px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
    })


def create_kpis(df):
    sono = df["Horas de Sono"].mean() if "Horas de Sono" in df.columns else None
    recovery = df["Pontuação de Recuperação"].mean() if "Pontuação de Recuperação" in df.columns else None
    hrv = df["VFC"].median() if "VFC" in df.columns else None
    treino = df["treinou"].mean() * 100 if "treinou" in df.columns else 0

    return html.Div([
        kpi_card("Sono Médio", f"{sono:.1f}h" if pd.notna(sono) else "-", CORES['sleep']),
        kpi_card("Recovery Médio", f"{recovery:.1f}" if pd.notna(recovery) else "-", CORES['recovery']),
        kpi_card("HRV Mediano", f"{hrv:.1f} ms" if pd.notna(hrv) else "-", CORES['hrv']),
        kpi_card("% Treino", f"{treino:.0f}%", CORES['strain'])
    ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap", "marginBottom": "30px"})


# ==================================================
# GRÁFICOS
# ==================================================

def users_by_gender(df):
    if "Gênero" not in df.columns:
        return html.Div("Dados de gênero não disponíveis", style={'color': CORES['text']})
    
    if "Usuário" in df.columns:
        user_col = "Usuário"
    elif "ID Usuário" in df.columns:
        user_col = "ID Usuário"
    elif "user_id" in df.columns:
        user_col = "user_id"
    else:
        return html.Div("Dados de usuário não disponíveis", style={'color': CORES['text']})
    
    resumo = df[[user_col, "Gênero"]].drop_duplicates().groupby("Gênero")[user_col].nunique().reset_index(name="usuários")
    
    fig = px.bar(resumo, x="usuários", y="Gênero", orientation="h", color="Gênero",
                 color_discrete_map={'Masculino': CORES['masculino'], 'Feminino': CORES['feminino']},
                 title="Distribuição de Usuários por Sexo")
    fig.update_layout(template=TEMPLATE, paper_bgcolor=CORES['background'], plot_bgcolor=CORES['background'],
                      font_color=CORES['text'], title_font_color=CORES['text'],
                      xaxis=dict(gridcolor=CORES['border'], title="Número de Usuários"),
                      yaxis=dict(title="Gênero", gridcolor=CORES['border']))
    fig.update_traces(marker_line_width=0, textposition='outside', texttemplate='%{x}')
    return dcc.Graph(figure=fig)


def sleep_by_age_gender(df):
    if "Horas de Sono" not in df.columns or "Gênero" not in df.columns:
        return html.Div("Dados de sono não disponíveis", style={'color': CORES['text']})
    
    temp = df.copy()
    temp["sleep_hours"] = temp["Horas de Sono"]
    temp["gender"] = temp["Gênero"]
    temp = add_age_group(temp)
    
    if "faixa_idade" not in temp.columns:
        return html.Div("Sem dados de idade para análise", style={'color': CORES['text']})
    
    faixas_traduzidas = {
        "Menos de 20": "Menos de 20",
        "20-29": "20-29 anos",
        "30-39": "30-39 anos",
        "40-49": "40-49 anos",
        "50-59": "50-59 anos",
        "60+": "60 anos ou mais"
    }
    temp["faixa_idade"] = temp["faixa_idade"].map(faixas_traduzidas)
    
    resumo = temp.groupby(["faixa_idade", "gender"])["sleep_hours"].mean().reset_index()
    return create_lollipop(resumo, "faixa_idade", "sleep_hours", "gender", 
                          "Média de Sono por Idade", "Faixa Etária", "Horas de Sono")


def recovery_by_age_gender(df):
    if "Pontuação de Recuperação" not in df.columns or "Gênero" not in df.columns:
        return html.Div("Dados de recuperação não disponíveis", style={'color': CORES['text']})
    
    temp = df.copy()
    temp["recovery_score"] = temp["Pontuação de Recuperação"]
    temp["gender"] = temp["Gênero"]
    temp = add_age_group(temp)
    
    if "faixa_idade" not in temp.columns:
        return html.Div("Sem dados de idade para análise", style={'color': CORES['text']})
    
    faixas_traduzidas = {
        "Menos de 20": "Menos de 20",
        "20-29": "20-29 anos",
        "30-39": "30-39 anos",
        "40-49": "40-49 anos",
        "50-59": "50-59 anos",
        "60+": "60 anos ou mais"
    }
    temp["faixa_idade"] = temp["faixa_idade"].map(faixas_traduzidas)
    
    resumo = temp.groupby(["faixa_idade", "gender"])["recovery_score"].mean().reset_index()
    return create_lollipop(resumo, "faixa_idade", "recovery_score", "gender", 
                          "Recuperação Média por Idade", "Faixa Etária", "Pontuação de Recuperação")


def hrv_by_age_gender(df):
    if "VFC" not in df.columns or "Gênero" not in df.columns:
        return html.Div("Dados de HRV não disponíveis", style={'color': CORES['text']})
    
    temp = df.copy()
    temp["hrv"] = temp["VFC"]
    temp["gender"] = temp["Gênero"]
    temp = add_age_group(temp)
    
    if "faixa_idade" not in temp.columns:
        return html.Div("Sem dados de idade para análise", style={'color': CORES['text']})
    
    faixas_traduzidas = {
        "Menos de 20": "Menos de 20",
        "20-29": "20-29 anos",
        "30-39": "30-39 anos",
        "40-49": "40-49 anos",
        "50-59": "50-59 anos",
        "60+": "60 anos ou mais"
    }
    temp["faixa_idade"] = temp["faixa_idade"].map(faixas_traduzidas)
    
    resumo = temp.groupby(["faixa_idade", "gender"])["hrv"].median().reset_index()
    return create_lollipop(resumo, "faixa_idade", "hrv", "gender", 
                          "HRV Mediano por Idade", "Faixa Etária", "HRV (ms)")


def fitness_level_distribution(df):
    """Distribuição do nível de condicionamento físico por usuário"""
    
    if "Nível de Condicionamento" not in df.columns:
        return html.Div("Dados de nível de condicionamento não disponíveis", style={'color': CORES['text']})
    
    if "Usuário" in df.columns:
        user_col = "Usuário"
    elif "ID Usuário" in df.columns:
        user_col = "ID Usuário"
    elif "user_id" in df.columns:
        user_col = "user_id"
    else:
        return html.Div("Dados de usuário não disponíveis", style={'color': CORES['text']})
    
    resumo = df[[user_col, "Nível de Condicionamento"]].drop_duplicates().groupby("Nível de Condicionamento")[user_col].nunique().reset_index()
    resumo.columns = ["nível", "quantidade"]
    
    niveis_ordem = ["Iniciante", "Intermediário", "Avançado", "Elite"]
    resumo["nível"] = pd.Categorical(resumo["nível"], categories=niveis_ordem, ordered=True)
    resumo = resumo.sort_values("nível")
    
    fig = px.bar(resumo, x="nível", y="quantidade", title="Distribuição de Usuários por Nível de Condicionamento",
                 text="quantidade", color="nível",
                 color_discrete_map={
                     "Iniciante": CORES['iniciante'],
                     "Intermediário": CORES['intermediario'],
                     "Avançado": CORES['avancado'],
                     "Elite": CORES['elite']
                 })
    fig.update_layout(template=TEMPLATE, paper_bgcolor=CORES['background'], plot_bgcolor=CORES['background'],
                      font_color=CORES['text'], title_font_color=CORES['text'],
                      xaxis=dict(gridcolor=CORES['border'], title="Nível de Condicionamento"),
                      yaxis=dict(gridcolor=CORES['border'], title="Número de Usuários"))
    fig.update_traces(textposition='outside')
    return dcc.Graph(figure=fig)


def day_strain_by_fitness(df):
    """Tensão do dia por nível de condicionamento"""
    
    if "Tensão do Dia" not in df.columns or "Nível de Condicionamento" not in df.columns:
        return html.Div("Dados de tensão ou condicionamento não disponíveis", style={'color': CORES['text']})
    
    resumo = df.groupby("Nível de Condicionamento")["Tensão do Dia"].mean().reset_index()
    
    niveis_ordem = ["Iniciante", "Intermediário", "Avançado", "Elite"]
    resumo["Nível de Condicionamento"] = pd.Categorical(resumo["Nível de Condicionamento"], categories=niveis_ordem, ordered=True)
    resumo = resumo.sort_values("Nível de Condicionamento")
    
    fig = px.bar(resumo, x="Nível de Condicionamento", y="Tensão do Dia", title="Tensão Média por Nível de Condicionamento",
                 text="Tensão do Dia", color="Nível de Condicionamento",
                 color_discrete_map={
                     "Iniciante": CORES['iniciante'],
                     "Intermediário": CORES['intermediario'],
                     "Avançado": CORES['avancado'],
                     "Elite": CORES['elite']
                 })
    fig.update_layout(template=TEMPLATE, paper_bgcolor=CORES['background'], plot_bgcolor=CORES['background'],
                      font_color=CORES['text'], title_font_color=CORES['text'],
                      xaxis=dict(gridcolor=CORES['border'], title="Nível de Condicionamento"),
                      yaxis=dict(gridcolor=CORES['border'], title="Tensão do Dia Média"))
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    return dcc.Graph(figure=fig)


def workout_days_count(df):
    """Contagem de dias treinados (dias com workout)"""
    
    if "Data" not in df.columns:
        return html.Div("Dados de data não disponíveis", style={'color': CORES['text']})
    
    if "treinou" in df.columns:
        dias_treinados = df[df["treinou"] == 1]["Data"].nunique()
    elif "Treino Realizado" in df.columns:
        dias_treinados = df[df["Treino Realizado"] == 1]["Data"].nunique()
    else:
        dias_treinados = df["Data"].nunique()
    
    total_dias = df["Data"].nunique()
    percentual = (dias_treinados / total_dias) * 100 if total_dias > 0 else 0
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = percentual,
        title = {"text": "Dias Treinados", "font": {"color": CORES['text']}},
        delta = {"reference": 80, "increasing": {"color": CORES['success']}},
        gauge = {
            "axis": {"range": [0, 100], "tickcolor": CORES['text']},
            "bar": {"color": CORES['accent']},
            "bgcolor": CORES['card_bg'],
            "borderwidth": 2,
            "bordercolor": CORES['text_secondary'],
            "steps": [
                {"range": [0, 50], "color": CORES['danger']},
                {"range": [50, 75], "color": CORES['warning']},
                {"range": [75, 100], "color": CORES['success']}
            ],
            "threshold": {
                "line": {"color": CORES['text'], "width": 4},
                "thickness": 0.75,
                "value": 80
            }
        }
    ))
    
    fig.update_layout(template=TEMPLATE, paper_bgcolor=CORES['background'], 
                      font_color=CORES['text'], height=300)
    
    info_card = html.Div([
        html.Div([
            html.H4(f"{dias_treinados}", style={'color': CORES['accent'], 'marginBottom': '5px'}),
            html.P("Dias com Treino", style={'color': CORES['text_secondary']})
        ], style={'textAlign': 'center', 'padding': '20px'}),
        html.Div([
            html.H4(f"{total_dias}", style={'color': CORES['sleep'], 'marginBottom': '5px'}),
            html.P("Total de Dias", style={'color': CORES['text_secondary']})
        ], style={'textAlign': 'center', 'padding': '20px'})
    ], style={'display': 'flex', 'gap': '20px', 'justifyContent': 'center', 'marginTop': '20px'})
    
    return html.Div([dcc.Graph(figure=fig), info_card])


def workout_time_distribution(df):
    """Distribuição de horários de treino"""
    
    if "Horário do Treino" not in df.columns:
        return html.Div("Dados de horário de treino não disponíveis", style={'color': CORES['text']})
    
    # Filtrar apenas treinos realizados
    if "treinou" in df.columns:
        df_treinos = df[df["treinou"] == 1]
    elif "Treino Realizado" in df.columns:
        df_treinos = df[df["Treino Realizado"] == 1]
    else:
        df_treinos = df
    
    resumo = df_treinos["Horário do Treino"].value_counts().reset_index()
    resumo.columns = ["horário", "quantidade"]
    
    horarios_ordem = ["Manhã", "Tarde", "Noite"]
    resumo["horário"] = pd.Categorical(resumo["horário"], categories=horarios_ordem, ordered=True)
    resumo = resumo.sort_values("horário")
    
    fig = px.bar(resumo, x="horário", y="quantidade", title="Distribuição de Horários de Treino",
                 text="quantidade", color="horário",
                 color_discrete_map={
                     "Manhã": CORES['manha'],
                     "Tarde": CORES['tarde'],
                     "Noite": CORES['noite']
                 })
    fig.update_layout(template=TEMPLATE, paper_bgcolor=CORES['background'], plot_bgcolor=CORES['background'],
                      font_color=CORES['text'], title_font_color=CORES['text'],
                      xaxis=dict(gridcolor=CORES['border'], title="Horário do Treino"),
                      yaxis=dict(gridcolor=CORES['border'], title="Quantidade de Treinos"))
    fig.update_traces(textposition='outside')
    return dcc.Graph(figure=fig)


# ==================================================
# LAYOUT FINAL
# ==================================================

def create_layout(df):
    if df is None or df.empty:
        return html.Div("Sem dados", style={"color": "white", "padding": "50px", "textAlign": "center"})

    try:
        df = preprocess_data(df)
    except Exception as e:
        return html.Div(f"Erro no processamento: {str(e)}", style={"color": "red", "padding": "50px", "textAlign": "center"})

    return html.Div([

        dbc.Button(
            "Voltar",
            href="/",
            color="dark",
            className="mb-4",
            style={"backgroundColor": "transparent", "border": f"1px solid {CORES['border']}", "color": CORES['text']}
        ),

        html.H1(
            "Análise de Dados de Wearables",
            style={'textAlign': 'center', 'color': CORES['text'], 'marginBottom': '30px'}
        ),

        create_kpis(df),

        html.Hr(style={'backgroundColor': CORES['border']}),

        users_by_gender(df),

        html.Hr(style={'backgroundColor': CORES['border']}),

        html.H4("Análise de Condicionamento", style={'color': CORES['text'], 'marginBottom': '20px'}),
        dbc.Row([
            dbc.Col(fitness_level_distribution(df), md=6),
            dbc.Col(day_strain_by_fitness(df), md=6),
        ], style={'marginBottom': '30px'}),

        html.Hr(style={'backgroundColor': CORES['border']}),

        html.H4("Análise de Treinos", style={'color': CORES['text'], 'marginBottom': '20px'}),
        dbc.Row([
            dbc.Col(workout_days_count(df), md=6),
            dbc.Col(workout_time_distribution(df), md=6),
        ], style={'marginBottom': '30px'}),

        html.Hr(style={'backgroundColor': CORES['border']}),

        html.H4("Análise por Faixa Etária", style={'color': CORES['text'], 'marginBottom': '20px'}),
        dbc.Row([
            dbc.Col(sleep_by_age_gender(df), md=6),
            dbc.Col(recovery_by_age_gender(df), md=6),
        ], style={'marginBottom': '30px'}),

        dbc.Row([
            dbc.Col(hrv_by_age_gender(df), md=12),
        ], style={'marginBottom': '30px'})

    ], style={
        "padding": "30px", "fontFamily": "Arial, sans-serif",
        "backgroundColor": CORES['background'], "color": CORES['text'], "minHeight": "100vh"
    })