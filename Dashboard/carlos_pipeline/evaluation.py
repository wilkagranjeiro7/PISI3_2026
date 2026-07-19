from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import (
    BASELINE_3VA_ACCURACY,
    BASELINE_3VA_CONTEXT,
    BASELINE_3VA_MODEL,
)
from .preprocessing import DadosPreparados


COR_PRINCIPAL = "#1F4E79"
COR_DESTAQUE = "#E17C05"
COR_NEUTRA = "#6B7280"
COR_GRADE = "#E5E7EB"


def _inteiro_pt(valor: float) -> str:
    return f"{valor:,.0f}".replace(",", ".")


def _decimal_pt(valor: float, casas: int = 2) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def _estilizar_eixo(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=COR_GRADE, linewidth=0.8)
    ax.set_axisbelow(True)


def avaliar_modelo(modelo, dados: DadosPreparados) -> tuple[dict, np.ndarray, np.ndarray]:
    previstos = modelo.predict(dados.x_teste)
    probabilidades = modelo.predict_proba(dados.x_teste)[:, 1]
    matriz = confusion_matrix(dados.y_teste, previstos, labels=[0, 1])

    metricas = {
        "modelo": "HistGradientBoostingClassifier",
        "accuracy": float(accuracy_score(dados.y_teste, previstos)),
        "balanced_accuracy": float(
            balanced_accuracy_score(dados.y_teste, previstos)
        ),
        "precision": float(precision_score(dados.y_teste, previstos)),
        "recall": float(recall_score(dados.y_teste, previstos)),
        "f1": float(f1_score(dados.y_teste, previstos)),
        "roc_auc": float(roc_auc_score(dados.y_teste, probabilidades)),
        "mcc": float(matthews_corrcoef(dados.y_teste, previstos)),
        "confusion_matrix": matriz.tolist(),
        "baseline_3va_model": BASELINE_3VA_MODEL,
        "baseline_3va_accuracy": BASELINE_3VA_ACCURACY,
        "baseline_3va_context": BASELINE_3VA_CONTEXT,
    }
    metricas["accuracy_difference_pp"] = (
        metricas["accuracy"] - BASELINE_3VA_ACCURACY
    ) * 100

    return metricas, previstos, probabilidades


def salvar_metricas(metricas: dict, destino: Path) -> None:
    destino.mkdir(parents=True, exist_ok=True)

    with (destino / "metricas.json").open("w", encoding="utf-8") as arquivo:
        json.dump(metricas, arquivo, ensure_ascii=False, indent=2)

    pd.DataFrame(
        [
            {"metrica": nome, "valor": valor}
            for nome, valor in metricas.items()
            if isinstance(valor, (int, float))
        ]
    ).to_csv(destino / "metricas.csv", index=False, encoding="utf-8-sig")


def grafico_balanceamento(dados: DadosPreparados, destino: Path) -> None:
    antes = dados.distribuicao_treino_antes
    depois = dados.distribuicao_treino_depois
    nomes = ["Recuperação baixa", "Boa recuperação"]
    x = np.arange(len(nomes))
    largura = 0.34

    fig, ax = plt.subplots(figsize=(8.1, 4.8))
    barras_antes = ax.bar(
        x - largura / 2,
        [antes.get(0, 0), antes.get(1, 0)],
        largura,
        label="Antes do SMOTE",
        color=COR_PRINCIPAL,
    )
    barras_depois = ax.bar(
        x + largura / 2,
        [depois.get(0, 0), depois.get(1, 0)],
        largura,
        label="Depois do SMOTE",
        color=COR_DESTAQUE,
    )
    ax.bar_label(
        barras_antes,
        labels=[_inteiro_pt(v) for v in barras_antes.datavalues],
        padding=3,
        fontsize=9,
    )
    ax.bar_label(
        barras_depois,
        labels=[_inteiro_pt(v) for v in barras_depois.datavalues],
        padding=3,
        fontsize=9,
    )
    ax.set_title("Distribuição das classes no conjunto de treino")
    ax.set_ylabel("Quantidade de registros")
    ax.set_xticks(x, nomes)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _inteiro_pt(v)))
    ax.legend(frameon=False, ncol=2, loc="upper center")
    _estilizar_eixo(ax)
    fig.tight_layout()
    fig.savefig(destino / "balanceamento_smote.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def grafico_matriz_confusao(metricas: dict, destino: Path) -> None:
    matriz = np.asarray(metricas["confusion_matrix"])
    percentuais = matriz / matriz.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.imshow(percentuais, cmap="Blues", vmin=0, vmax=1)

    for linha in range(2):
        for coluna in range(2):
            ax.text(
                coluna,
                linha,
                f"{_inteiro_pt(matriz[linha, coluna])}\n"
                f"({_decimal_pt(percentuais[linha, coluna] * 100, 1)}%)",
                ha="center",
                va="center",
                color="white" if percentuais[linha, coluna] > 0.5 else "#111827",
                fontsize=12,
                fontweight="bold",
            )

    ax.set_title("Matriz de confusão no conjunto de teste")
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_xticks([0, 1], ["Baixa", "Boa"])
    ax.set_yticks([0, 1], ["Baixa", "Boa"])
    fig.tight_layout()
    fig.savefig(destino / "matriz_confusao.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def grafico_roc(y_real, probabilidades: np.ndarray, metricas: dict, destino: Path) -> None:
    falso_positivo, verdadeiro_positivo, _ = roc_curve(y_real, probabilidades)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot(
        falso_positivo,
        verdadeiro_positivo,
        color=COR_DESTAQUE,
        linewidth=2.5,
        label=f"HistGradientBoosting (AUC = {_decimal_pt(metricas['roc_auc'], 3)})",
    )
    ax.plot([0, 1], [0, 1], "--", color=COR_NEUTRA, label="Classificador aleatório")
    ax.set_title("Curva ROC no conjunto de teste")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")
    ax.legend(frameon=False, loc="lower right")
    _estilizar_eixo(ax)
    fig.tight_layout()
    fig.savefig(destino / "curva_roc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def grafico_comparacao_3va(metricas: dict, destino: Path) -> None:
    nomes = [
        f"Resultado relatado na 3VA\n{BASELINE_3VA_MODEL}",
        "Pipeline final\nHistGradientBoosting",
    ]
    valores = [BASELINE_3VA_ACCURACY * 100, metricas["accuracy"] * 100]
    cores = [COR_NEUTRA, COR_DESTAQUE]

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    barras = ax.bar(nomes, valores, color=cores, width=0.55)
    ax.bar_label(
        barras,
        labels=[f"{_decimal_pt(valor, 2)}%" for valor in valores],
        padding=4,
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Acurácia (%)")
    ax.set_title("Comparação descritiva de acurácia", fontsize=15)
    ax.text(
        0.5,
        1.01,
        "Diferença observada: "
        f"{_decimal_pt(metricas['accuracy_difference_pp'], 2)} p.p.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
    )
    _estilizar_eixo(ax)
    fig.text(
        0.5,
        0.02,
        "Protocolos de avaliação diferentes; valores usados como referência descritiva.",
        ha="center",
        fontsize=9,
        color=COR_NEUTRA,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(destino / "comparacao_3va.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
