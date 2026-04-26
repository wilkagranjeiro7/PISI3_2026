from dash import html, dcc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ==================================================
# 🎨 CONFIGURAÇÕES DE DESIGN
# ==================================================
# Paleta (Teal e Laranja queimado)
COLOR_MAP = {'Male': '#2A9D8F', 'Female': '#E76F51'}
TEMPLATE = 'plotly_white'

# ==================================================
# 🔧 FUNÇÕES AUXILIARES E PREPROCESSAMENTO
# ==================================================

def convert_to_hours(series):
    """Detecta se a coluna está em horas, minutos ou segundos e converte para horas."""
    serie = pd.to_numeric(series, errors="coerce")
    validos = serie.dropna()
    if validos.empty: return serie
    ref = validos.quantile(0.95)
    # Se for horas (ref <= 24), mantém. Se for minutos ou segundos, divide.
    if ref <= 24: pass
    elif ref <= 1500: serie = serie / 60
    else: serie = serie / 3600
    return serie

def clean_hrv(series):
    """Corrige valores extremos de HRV e ajusta escalas inconsistentes."""
    serie = pd.to_numeric(series, errors="coerce")
    serie.loc[serie > 5000] = pd.NA
    validos = serie.dropna()
    if validos.empty: return serie
    ref = validos.quantile(0.95)
    # Ajuste de escala (x100 ou x10) para normalizar dados
    if ref > 1000: serie = serie / 100
    elif ref > 250: serie = serie / 10
    serie.loc[(serie < 10) | (serie > 250)] = pd.NA
    return serie

def preprocess_data(df):
    """Limpeza robusta de outliers e tratamento de nulos."""
    df = df.copy()
    if "date" in df.columns: df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["treinou"] = df["workout_time_of_day"].notna().astype(int) if "workout_time_of_day" in df.columns else 0
    
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df.loc[(df["age"] < 10) | (df["age"] > 100), "age"] = pd.NA
        
    if "recovery_score" in df.columns:
        df["recovery_score"] = pd.to_numeric(df["recovery_score"], errors="coerce")
        df.loc[(df["recovery_score"] < 0) | (df["recovery_score"] > 100), "recovery_score"] = pd.NA
        
    if "hrv" in df.columns: df["hrv"] = clean_hrv(df["hrv"])
    
    if "sleep_hours" in df.columns:
        df["sleep_hours"] = pd.to_numeric(df["sleep_hours"], errors="coerce")
        df.loc[(df["sleep_hours"] < 2) | (df["sleep_hours"] > 16), "sleep_hours"] = pd.NA

    # Tratamento de fases do sono
    for col in ["light_sleep_hours", "deep_sleep_hours", "rem_sleep_hours"]:
        if col in df.columns:
            df[col] = convert_to_hours(df[col])
            df.loc[(df[col] < 0) | (df[col] > 12), col] = pd.NA
            if "sleep_hours" in df.columns:
                df.loc[df[col] > df["sleep_hours"], col] = pd.NA
    return df

def add_age_group(df):
    """Cria faixas etárias para agrupamento."""
    df = df.copy()
    bins = [0, 20, 30, 40, 50, 60, 100]
    labels = ["<20", "20-29", "30-39", "40-49", "50-59", "60+"]
    df["faixa_idade"] = pd.cut(df["age"], bins=bins, labels=labels)
    return df

# ==================================================
# 🍬 FUNÇÃO DE VISUALIZAÇÃO (LOLLIPOP)
# ==================================================

def create_lollipop(df, x_col, y_col, color_col, title, y_axis_title):
    """Gera gráfico Lollipop, fugindo do visual trivial de colunas."""
    fig = go.Figure()
    for cat in df[color_col].unique():
        subset = df[df[color_col] == cat]
        fig.add_trace(go.Scatter(
            x=subset[x_col], y=subset[y_col], mode='markers+lines',
            name=cat, marker=dict(size=12, color=COLOR_MAP.get(cat, '#888')),
            line=dict(width=2, color=COLOR_MAP.get(cat, '#888'))
        ))
    fig.update_layout(title=title, template=TEMPLATE, yaxis_title=y_axis_title)
    return dcc.Graph(figure=fig)

# ==================================================
# 📊 KPIS E COMPONENTES
# ==================================================

def kpi_card(title, value):
    return html.Div([html.H5(title), html.H2(value)], 
        style={"background": "#f8f9fa", "padding": "20px", "borderRadius": "10px", "textAlign": "center", "flex": "1"})

def create_kpis(df):
    sono = df["sleep_hours"].mean() if "sleep_hours" in df.columns else None
    recovery = df["recovery_score"].mean() if "recovery_score" in df.columns else None
    hrv = df["hrv"].median() if "hrv" in df.columns else None
    treino = df["treinou"].mean() * 100
    return html.Div([
        kpi_card("Sono Médio", f"{sono:.1f}h" if pd.notna(sono) else "-"),
        kpi_card("Recovery Médio", f"{recovery:.1f}" if pd.notna(recovery) else "-"),
        kpi_card("HRV Mediano", f"{hrv:.1f} ms" if pd.notna(hrv) else "-"),
        kpi_card("% Treino", f"{treino:.0f}%")
    ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap", "marginBottom": "30px"})

# ==================================================
# 📉 GRÁFICOS
# ==================================================

def users_by_gender(df):
    resumo = df[["user_id", "gender"]].drop_duplicates().groupby("gender")["user_id"].nunique().reset_index(name="usuarios")
    fig = px.bar(resumo, x="usuarios", y="gender", orientation="h", color="gender", 
                 color_discrete_map=COLOR_MAP, title="Distribuição de Usuários por Gênero")
    fig.update_layout(template=TEMPLATE)
    return dcc.Graph(figure=fig)

def hrv_by_age_gender(df):
    temp = add_age_group(df)
    resumo = temp.groupby(["faixa_idade", "gender"])["hrv"].median().reset_index()
    return create_lollipop(resumo, "faixa_idade", "hrv", "gender", "HRV Mediano por Idade", "HRV (ms)")

def sleep_by_age_gender(df):
    temp = add_age_group(df)
    resumo = temp.groupby(["faixa_idade", "gender"])["sleep_hours"].mean().reset_index()
    return create_lollipop(resumo, "faixa_idade", "sleep_hours", "gender", "Média de Sono por Idade", "Horas")

def recovery_by_age_gender(df):
    temp = add_age_group(df)
    resumo = temp.groupby(["faixa_idade", "gender"])["recovery_score"].mean().reset_index()
    return create_lollipop(resumo, "faixa_idade", "recovery_score", "gender", "Recovery Médio por Idade", "Score")

# ==================================================
# 🧱 LAYOUT FINAL
# ==================================================

def create_layout(df):
    df = preprocess_data(df)

    return html.Div([
        # Título coerente com dados biométricos/wearables
        html.H1("📊 Análise de Dados de Wearables: Sono, Recovery e Fitness", style={'textAlign': 'center'}),
        
        # Caixa explicativa para responder ao "Target Audience"
        html.Div([
            html.H3("Target Audience:"),
            html.P("Este dashboard foi desenvolvido para analisar padrões de saúde e biométricos coletados por dispositivos vestíveis (smartwatches), auxiliando no monitoramento de recuperação e performance.")
        ], style={'background': '#f0f4f8', 'padding': '15px', 'borderRadius': '10px', 'marginBottom': '20px'}),

        create_kpis(df),

        html.Hr(),
        users_by_gender(df),

        html.Hr(),
        # Agrupamento para layouts lado a lado
        html.Div([
            html.Div([sleep_by_age_gender(df)], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([recovery_by_age_gender(df)], style={'width': '48%', 'display': 'inline-block'})
        ]),

        html.Hr(),
        hrv_by_age_gender(df)

    ], style={"padding": "30px", "fontFamily": "sans-serif"})