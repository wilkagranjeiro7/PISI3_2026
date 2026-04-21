from dash import html, dcc
import plotly.express as px
import pandas as pd


# =====================================
# 🔧 PREPROCESSAMENTO
# =====================================

def preprocess_data(df):
    df = df.copy()

    if 'workout_time_of_day' in df.columns:
        df['treinou'] = df['workout_time_of_day'].notna().astype(int)
    else:
        df['treinou'] = 0

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')

    if 'sleep_hours' in df.columns:
        df['sleep_hours'] = pd.to_numeric(
            df['sleep_hours'],
            errors='coerce'
        )

    if 'recovery_score' in df.columns:
        df['recovery_score'] = pd.to_numeric(
            df['recovery_score'],
            errors='coerce'
        )

    if 'hrv' in df.columns:
        df['hrv'] = pd.to_numeric(
            df['hrv'],
            errors='coerce'
        )

    return df


# =====================================
# 📊 KPI CARD
# =====================================

def kpi_card(title, value):
    return html.Div([
        html.H5(title),
        html.H2(value)
    ], style={
        "background": "#f8f9fa",
        "padding": "20px",
        "borderRadius": "10px",
        "textAlign": "center",
        "flex": "1",
        "minWidth": "220px"
    })


# =====================================
# 📊 KPIS
# =====================================

def create_kpis(df):
    return html.Div([

        kpi_card(
            "Recovery Médio",
            f"{df['recovery_score'].mean():.1f}"
            if 'recovery_score' in df.columns else "-"
        ),

        kpi_card(
            "Sono Médio",
            f"{df['sleep_hours'].mean():.1f}h"
            if 'sleep_hours' in df.columns else "-"
        ),

        kpi_card(
            "HRV Médio",
            f"{df['hrv'].mean():.1f}"
            if 'hrv' in df.columns else "-"
        ),

        kpi_card(
            "% Treino",
            f"{df['treinou'].mean()*100:.0f}%"
        )

    ], style={
        "display": "flex",
        "gap": "20px",
        "flexWrap": "wrap",
        "marginBottom": "30px"
    })


# =====================================
# 👥 USUÁRIOS POR GÊNERO
# =====================================

def users_by_gender(df):
    if not {'user_id', 'gender'}.issubset(df.columns):
        return html.Div()

    resumo = (
        df[['user_id', 'gender']]
        .drop_duplicates()
        .groupby('gender')
        .size()
        .reset_index(name='usuarios')
    )

    fig = px.pie(
        resumo,
        names='gender',
        values='usuarios',
        title='Usuários por Gênero'
    )

    fig.update_layout(template='plotly_white')

    return dcc.Graph(figure=fig)


# =====================================
# 🎂 IDADE POR GÊNERO
# =====================================

def age_by_gender(df):
    if not {'age', 'gender'}.issubset(df.columns):
        return html.Div()

    fig = px.box(
        df,
        x='gender',
        y='age',
        color='gender',
        title='Distribuição de Idade por Gênero'
    )

    fig.update_layout(template='plotly_white')

    return dcc.Graph(figure=fig)


# =====================================
# 🏃 ATIVIDADES POR GÊNERO
# =====================================

def sport_by_gender(df):
    if not {'gender', 'primary_sport'}.issubset(df.columns):
        return html.Div()

    temp = df.copy()

    temp['primary_sport'] = (
        temp['primary_sport']
        .astype(str)
        .str.strip()
        .str.title()
    )

    resumo = (
        temp.groupby(['gender', 'primary_sport'])
        .size()
        .reset_index(name='quantidade')
    )

    fig = px.bar(
        resumo,
        x='primary_sport',
        y='quantidade',
        color='gender',
        barmode='group',
        text_auto=True,
        title='Atividades por Gênero'
    )

    fig.update_layout(
        template='plotly_white',
        xaxis_tickangle=-35,
        height=500
    )

    return dcc.Graph(figure=fig)


# =====================================
# 😴 SONO
# =====================================

def sleep_by_age_gender(df):
    if not {'age', 'gender', 'sleep_hours'}.issubset(df.columns):
        return html.Div()

    temp = df.copy()

    bins = [0, 20, 30, 40, 50, 60, 100]
    labels = ['<20', '20-29', '30-39', '40-49', '50-59', '60+']

    temp['faixa_idade'] = pd.cut(
        temp['age'],
        bins=bins,
        labels=labels
    )

    resumo = (
        temp.groupby(
            ['faixa_idade', 'gender']
        )['sleep_hours']
        .mean()
        .reset_index()
    )

    fig = px.bar(
        resumo,
        x='faixa_idade',
        y='sleep_hours',
        color='gender',
        barmode='group',
        text_auto='.1f',
        title='Sono Médio por Faixa Etária e Gênero'
    )

    fig.update_layout(template='plotly_white')

    return dcc.Graph(figure=fig)


# =====================================
# 💚 RECOVERY
# =====================================

def recovery_by_age_gender(df):
    if not {'age', 'gender', 'recovery_score'}.issubset(df.columns):
        return html.Div()

    temp = df.copy()

    bins = [0, 20, 30, 40, 50, 60, 100]
    labels = ['<20', '20-29', '30-39', '40-49', '50-59', '60+']

    temp['faixa_idade'] = pd.cut(
        temp['age'],
        bins=bins,
        labels=labels
    )

    resumo = (
        temp.groupby(
            ['faixa_idade', 'gender']
        )['recovery_score']
        .mean()
        .reset_index()
    )

    fig = px.bar(
        resumo,
        x='faixa_idade',
        y='recovery_score',
        color='gender',
        barmode='group',
        text_auto='.1f',
        title='Recovery Médio por Faixa Etária e Gênero'
    )

    fig.update_layout(template='plotly_white')

    return dcc.Graph(figure=fig)


# =====================================
# 🏋 FITNESS LEVEL
# =====================================

def fitness_by_age_gender(df):
    if not {'age', 'gender', 'fitness_level'}.issubset(df.columns):
        return html.Div()

    temp = df.copy()

    bins = [0, 20, 30, 40, 50, 60, 100]
    labels = ['<20', '20-29', '30-39', '40-49', '50-59', '60+']

    temp['faixa_idade'] = pd.cut(
        temp['age'],
        bins=bins,
        labels=labels
    )

    temp['fitness_level'] = (
        temp['fitness_level']
        .astype(str)
        .str.strip()
        .str.title()
    )

    resumo = (
        temp.groupby(
            ['faixa_idade', 'gender', 'fitness_level']
        )
        .size()
        .reset_index(name='quantidade')
    )

    fig = px.bar(
        resumo,
        x='faixa_idade',
        y='quantidade',
        color='fitness_level',
        facet_col='gender',
        barmode='group',
        text_auto=True,
        title='Fitness Level por Idade e Gênero'
    )

    fig.update_layout(
        template='plotly_white',
        height=550
    )

    return dcc.Graph(figure=fig)


# =====================================
# 💡 INSIGHTS
# =====================================

def generate_insights(df):
    insights = []

    if {'hrv', 'recovery_score'}.issubset(df.columns):
        corr = df['hrv'].corr(df['recovery_score'])
        insights.append(
            html.Li(f"Correlação HRV x Recovery: {corr:.2f}")
        )

    if {'sleep_hours', 'recovery_score'}.issubset(df.columns):
        corr = df['sleep_hours'].corr(df['recovery_score'])
        insights.append(
            html.Li(f"Correlação Sono x Recovery: {corr:.2f}")
        )

    return html.Div([
        html.H4("Insights Automáticos"),
        html.Ul(insights)
    ])


# =====================================
# 🧱 LAYOUT FINAL
# =====================================

def create_layout(df):
    df = preprocess_data(df)

    return html.Div([

        html.H1("📊 EDA - Análise Exploratória Completa"),

        create_kpis(df),

        html.Hr(),

        html.H2("👥 Perfil Demográfico"),
        users_by_gender(df),
        age_by_gender(df),

        html.Hr(),

        html.H2("🏃 Atividades"),
        sport_by_gender(df),

        html.Hr(),

        html.H2("😴 Sono"),
        sleep_by_age_gender(df),

        html.Hr(),

        html.H2("💚 Recovery"),
        recovery_by_age_gender(df),

        html.Hr(),

        html.H2("🏋 Fitness Level"),
        fitness_by_age_gender(df),

        html.Hr(),

        generate_insights(df)

    ], style={
        "padding": "20px"
    })