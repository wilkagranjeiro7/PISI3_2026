# pages/plots.py
from dash import html, dcc, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd

from data_loader import data_manager


# ==================================================
# CORES PADRÃO (via DataManager)
# ==================================================

CORES = data_manager.get_cores()
TEMPLATE = 'plotly_dark'

# Mapeamento de cores para categorias (usando a paleta padronizada)
COLOR_MAP = {
    # Gênero
    'Masculino': CORES['masculino'],
    'Feminino': CORES['feminino'],
    # Nível de condicionamento
    'Iniciante': CORES['iniciante'],
    'Intermediário': CORES['intermediario'],
    'Avançado': CORES['avancado'],
    'Elite': CORES['elite'],
    # Horários
    'Manhã': CORES['manha'],
    'Tarde': CORES['tarde'],
    'Noite': CORES['noite'],
}

DEFAULT_COLORS = CORES['chart_colors']


def create_layout(df):
    """Layout com dbc.Select"""
    
    # Identificar colunas numéricas
    numeric_cols = [c for c in df.columns if df[c].dtype in ['int64', 'float64']]
    
    # Criar opções traduzidas para os selects
    col_options = [
        {'label': data_manager.traduzir_coluna(c), 'value': c} 
        for c in numeric_cols
    ]
    
    # Opções para cores (categóricas)
    categorical_cols = [c for c in ['fitness_level', 'gender', 'primary_sport', 'activity_type'] if c in df.columns]
    
    color_options = [{'label': 'Nenhum', 'value': 'none'}]
    color_options.extend([
        {'label': data_manager.traduzir_coluna(c), 'value': c}
        for c in categorical_cols
    ])
    
    # Estilo padrão para os selects
    SELECT_STYLE = {
        'backgroundColor': CORES['card_bg'],
        'color': CORES['text'],
        'border': f'1px solid {CORES["border"]}',
        'borderRadius': '4px'
    }
    
    return html.Div([
        # Botão voltar (fixo no canto superior esquerdo)
        html.Div([
            dbc.Button(
                "← Voltar",
                href="/",
                color="light",
                size="sm",
                style={'backgroundColor': 'transparent', 'border': f'1px solid {CORES["border"]}', 
                       'color': CORES['text']}
            )
        ], style={'position': 'fixed', 'top': '20px', 'left': '20px', 'zIndex': '1000'}),
        
        # Conteúdo principal
        html.Div([
            # Painel esquerdo - configurações (FIXO)
            html.Div([
                html.H3("Visualizações", style={'fontWeight': 'normal', 'marginBottom': '30px', 'color': CORES['text']}),
                
                # Tipo de gráfico
                html.Div([
                    html.Label("TIPO DE GRÁFICO", style={'color': CORES['text_secondary'], 'fontSize': '12px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                    html.Div([
                        dbc.Select(
                            id='plot-type',
                            options=[
                                {'label': 'Dispersão (Scatter)', 'value': 'scatter'},
                                {'label': 'Histograma', 'value': 'histogram'},
                                {'label': 'Boxplot', 'value': 'box'},
                                {'label': 'Violin Plot', 'value': 'violin'},
                                {'label': 'Barras (Bar)', 'value': 'bar'},
                                {'label': 'Densidade', 'value': 'density'}
                            ],
                            value='scatter',
                            style=SELECT_STYLE
                        )
                    ], style={'marginTop': '10px'})
                ], style={'marginBottom': '30px'}),
                
                # Eixo X
                html.Div([
                    html.Label("EIXO X", style={'color': CORES['text_secondary'], 'fontSize': '12px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                    html.Div([
                        dbc.Select(
                            id='plot-x',
                            options=col_options,
                            value=numeric_cols[0] if numeric_cols else None,
                            style=SELECT_STYLE
                        )
                    ], style={'marginTop': '10px'})
                ], style={'marginBottom': '30px'}),
                
                # Eixo Y (para gráficos que precisam)
                html.Div(id='plot-y-container', children=[
                    html.Label("EIXO Y", style={'color': CORES['text_secondary'], 'fontSize': '12px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                    html.Div([
                        dbc.Select(
                            id='plot-y',
                            options=col_options,
                            value=numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0] if numeric_cols else None,
                            style=SELECT_STYLE
                        )
                    ], style={'marginTop': '10px'})
                ], style={'marginBottom': '30px'}),
                
                # Cor (Agrupamento)
                html.Div([
                    html.Label("AGRUPAR POR", style={'color': CORES['text_secondary'], 'fontSize': '12px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                    html.Div([
                        dbc.Select(
                            id='plot-color',
                            options=color_options,
                            value='none',
                            style=SELECT_STYLE
                        )
                    ], style={'marginTop': '10px'})
                ], style={'marginBottom': '30px'}),
                
                # Controles extras (apenas para scatter)
                html.Div(id='extra-controls', children=[
                    html.Div([
                        html.Label("TAMANHO DOS PONTOS", style={'color': CORES['text_secondary'], 'fontSize': '12px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                        html.Div([
                            dcc.Slider(
                                id='point-size',
                                min=3,
                                max=15,
                                step=1,
                                value=8,
                                marks={i: str(i) for i in range(3, 16, 3)}
                            )
                        ], style={'marginTop': '10px'})
                    ], style={'marginBottom': '20px'}),
                    
                    html.Div([
                        html.Label("TRANSPARÊNCIA", style={'color': CORES['text_secondary'], 'fontSize': '12px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                        html.Div([
                            dcc.Slider(
                                id='opacity',
                                min=0.3,
                                max=1.0,
                                step=0.1,
                                value=0.7,
                                marks={i/10: str(i/10) for i in range(3, 11, 2)}
                            )
                        ], style={'marginTop': '10px'})
                    ], style={'marginBottom': '20px'}),
                ]),
                
            ], style={
                'position': 'fixed', 
                'width': '300px', 
                'padding': '80px 25px 20px 25px',
                'borderRight': f'1px solid {CORES["border"]}',
                'height': '100vh',
                'overflowY': 'auto',
                'backgroundColor': CORES['background']
            }),
            
            # Painel direito - gráfico
            html.Div([
                html.Div(id='plot-grafico', children=[
                    html.Div([
                        html.P("Selecione as variáveis para visualizar o gráfico", 
                              style={'color': CORES['text_secondary'], 'textAlign': 'center', 'marginTop': '50px'})
                    ])
                ])
            ], style={'marginLeft': '320px', 'padding': '20px', 'minHeight': '100vh'})
            
        ])
        
    ], style={'backgroundColor': CORES['background'], 'minHeight': '100vh', 'color': CORES['text']})


# ==================================================
# CALLBACKS
# ==================================================

@callback(
    Output('plot-y-container', 'style'),
    Input('plot-type', 'value')
)
def toggle_y_axis(plot_type):
    """Mostra ou esconde o eixo Y dependendo do tipo de gráfico"""
    if plot_type in ['histogram', 'box', 'violin']:
        return {'display': 'none'}
    return {'marginBottom': '30px'}


@callback(
    Output('extra-controls', 'style'),
    Input('plot-type', 'value')
)
def toggle_extra_controls(plot_type):
    """Mostra controles extras apenas para scatter"""
    if plot_type == 'scatter':
        return {'display': 'block', 'marginTop': '20px'}
    return {'display': 'none'}


@callback(
    Output('plot-grafico', 'children'),
    Input('plot-type', 'value'),
    Input('plot-x', 'value'),
    Input('plot-y', 'value'),
    Input('plot-color', 'value'),
    Input('point-size', 'value'),
    Input('opacity', 'value')
)
def update_plot(plot_type, x_col, y_col, color_col, point_size, opacity):
    """Gera o gráfico baseado nas seleções"""
    
    df = data_manager.get_clean_df()
    
    if df is None or df.empty:
        return html.Div("❌ Dados não disponíveis", style={'color': CORES['danger'], 'textAlign': 'center', 'padding': 50})
    
    if x_col is None:
        return html.Div("⚠️ Selecione uma coluna para o eixo X", style={'color': CORES['warning'], 'textAlign': 'center', 'padding': 50})
    
    if plot_type not in ['histogram', 'box', 'violin'] and y_col is None:
        return html.Div("⚠️ Selecione uma coluna para o eixo Y", style={'color': CORES['warning'], 'textAlign': 'center', 'padding': 50})
    
    color = None if color_col == 'none' else color_col
    
    # Traduzir labels
    x_label = data_manager.traduzir_coluna(x_col)
    y_label = data_manager.traduzir_coluna(y_col) if y_col else None
    
    # Preparar dados
    df_plot = df.copy()
    
    # Traduzir valores categóricos
    color_label = None
    
    if color and color in df_plot.columns:
        if df_plot[color].dtype == 'object':
            # Traduzir valores
            df_plot[color] = df_plot[color].apply(
                lambda x: data_manager.VALUE_TRANSLATIONS.get(str(x).lower(), x) if pd.notna(x) else x
            )
        color_label = data_manager.traduzir_coluna(color)
    
    try:
        # Criar gráfico - USANDO COLOR_MAP DIRETAMENTE
        if plot_type == 'scatter':
            fig = px.scatter(
                df_plot, x=x_col, y=y_col, color=color,
                title=f'{x_label} vs {y_label}',
                opacity=opacity,
                color_discrete_map=COLOR_MAP,
                labels={x_col: x_label, y_col: y_label, color: color_label if color_label else x_col}
            )
            fig.update_traces(marker=dict(size=point_size))
            if not color:
                fig.update_traces(marker=dict(color=CORES['accent']))
                
        elif plot_type == 'histogram':
            fig = px.histogram(
                df_plot, x=x_col, color=color, nbins=30,
                title=f'Distribuição de {x_label}',
                color_discrete_map=COLOR_MAP,
                labels={x_col: x_label, color: color_label if color_label else x_col}
            )
            
        elif plot_type == 'box':
            fig = px.box(
                df_plot, y=x_col, color=color,
                title=f'Boxplot de {x_label}',
                color_discrete_map=COLOR_MAP,
                labels={x_col: x_label, color: color_label if color_label else x_col}
            )
            
        elif plot_type == 'violin':
            fig = px.violin(
                df_plot, y=x_col, color=color, box=True,
                title=f'Violin Plot de {x_label}',
                color_discrete_map=COLOR_MAP,
                labels={x_col: x_label, color: color_label if color_label else x_col}
            )
            
        elif plot_type == 'density':
            fig = px.density_contour(
                df_plot, x=x_col, y=y_col,
                title=f'Densidade de {x_label} vs {y_label}',
                color_discrete_map=COLOR_MAP,
                labels={x_col: x_label, y_col: y_label}
            )
            if color:
                fig.update_traces(contours_coloring='fill', colorscale='Viridis')
            
        else:  # bar
            df_copy = df_plot.copy()
            if df_copy[x_col].nunique() > 20:
                df_copy[f'{x_col}_group'] = pd.cut(df_copy[x_col], bins=10)
                agg_col = f'{x_col}_group'
            else:
                agg_col = x_col
            agg_df = df_copy.groupby(agg_col)[y_col].mean().reset_index()
            if agg_col != x_col:
                agg_df[agg_col] = agg_df[agg_col].astype(str)
            fig = px.bar(
                agg_df, x=agg_col, y=y_col,
                title=f'Média de {y_label} por {x_label}',
                color_discrete_sequence=[CORES['accent']],
                text_auto='.2f',
                labels={agg_col: x_label, y_col: y_label}
            )
        
        # Aplicar tema escuro com cores padronizadas
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=CORES['card_bg'],
            plot_bgcolor=CORES['card_bg'],
            font_color=CORES['text'],
            title_font_color=CORES['text'],
            title_x=0.5,
            height=600,
            xaxis=dict(gridcolor=CORES['border'], title_font_color=CORES['text_secondary']),
            yaxis=dict(gridcolor=CORES['border'], title_font_color=CORES['text_secondary']),
            legend=dict(bgcolor=CORES['card_bg'], bordercolor=CORES['border'])
        )
        
        return dcc.Graph(figure=fig, config={'displayModeBar': True, 'displaylogo': False})
        
    except Exception as e:
        import traceback
        print(f"Erro detalhado: {traceback.format_exc()}")
        return html.Div(f"❌ Erro ao gerar gráfico: {str(e)}", style={'color': CORES['danger'], 'textAlign': 'center', 'padding': 50})