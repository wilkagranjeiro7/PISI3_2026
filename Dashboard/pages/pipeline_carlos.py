from __future__ import annotations

import base64
import json
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import html

RESULTADOS = Path(__file__).resolve().parents[1] / "resultados_carlos"


def _imagem(nome: str, titulo: str):
    caminho = RESULTADOS / nome
    if not caminho.exists():
        return html.Div(f"Imagem ainda não gerada: {nome}", className="text-warning")

    conteudo = base64.b64encode(caminho.read_bytes()).decode("ascii")
    return dbc.Card(
        [
            dbc.CardHeader(titulo),
            dbc.CardBody(
                html.Img(
                    src=f"data:image/png;base64,{conteudo}",
                    style={"width": "100%", "borderRadius": "8px"},
                )
            ),
        ],
        className="h-100",
    )


def _cartao_metrica(titulo: str, valor: str, cor: str):
    return dbc.Card(
        dbc.CardBody(
            [
                html.P(titulo, className="text-secondary mb-1"),
                html.H3(valor, style={"color": cor, "marginBottom": 0}),
            ]
        ),
        className="h-100",
    )


def create_layout(_df=None):
    metricas_path = RESULTADOS / "metricas.json"

    if not metricas_path.exists():
        return dbc.Container(
            [
                dbc.Button("← Voltar", href="/", color="secondary", className="mt-4"),
                html.H2("Pipeline individual - Carlos Jonathan", className="mt-4"),
                dbc.Alert(
                    "Execute primeiro: python Dashboard/run_pipeline_carlos.py",
                    color="warning",
                    className="mt-3",
                ),
            ],
            fluid=True,
        )

    metricas = json.loads(metricas_path.read_text(encoding="utf-8"))

    return dbc.Container(
        [
            dbc.Button("← Voltar", href="/", color="secondary", className="mt-4"),
            html.H2("Pipeline individual - Carlos Jonathan", className="mt-4"),
            html.P(
                "Target: boa recuperação quando recovery_score > 66. "
                "O SMOTE foi aplicado somente no treino; o teste permaneceu com a distribuição real.",
                className="text-secondary",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        _cartao_metrica(
                            "Acurácia", f"{metricas['accuracy']:.2%}", "#F28C1B"
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        _cartao_metrica("F1", f"{metricas['f1']:.2%}", "#38BDF8"),
                        md=3,
                    ),
                    dbc.Col(
                        _cartao_metrica(
                            "AUC-ROC", f"{metricas['roc_auc']:.3f}", "#34D399"
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        _cartao_metrica(
                            "Diferença para a referência da 3VA",
                            f"{metricas['accuracy_difference_pp']:.2f} p.p.",
                            "#A78BFA",
                        ),
                        md=3,
                    ),
                ],
                className="g-3 mt-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        _imagem(
                            "comparacao_3va.png",
                            "Comparação descritiva com a 3VA",
                        ),
                        md=6,
                    ),
                    dbc.Col(_imagem("balanceamento_smote.png", "Balanceamento"), md=6),
                ],
                className="g-3 mt-2",
            ),
            dbc.Row(
                [
                    dbc.Col(_imagem("matriz_confusao.png", "Matriz de confusão"), md=6),
                    dbc.Col(_imagem("curva_roc.png", "Curva ROC"), md=6),
                ],
                className="g-3 mt-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        _imagem("shap_importancia_global.png", "XAI - importância global SHAP"),
                        md=6,
                    ),
                    dbc.Col(_imagem("shap_resumo.png", "XAI - direção do impacto"), md=6),
                ],
                className="g-3 my-3",
            ),
        ],
        fluid=True,
        style={"maxWidth": "1500px"},
    )
