from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .config import FEATURE_LABELS, RANDOM_STATE


COR_DESTAQUE = "#E17C05"


def _decimal_pt(valor: float, casas: int = 3) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def gerar_xai(modelo, x_teste: pd.DataFrame, destino: Path, amostra=800) -> pd.DataFrame:
    """Gera explicações globais SHAP a partir de uma amostra reproduzível do teste."""
    quantidade = min(amostra, len(x_teste))
    x_amostra = x_teste.sample(quantidade, random_state=RANDOM_STATE)
    explicador = shap.Explainer(modelo)
    explicacoes = explicador(x_amostra)

    valores = explicacoes.values
    if valores.ndim == 3:
        valores = valores[:, :, 1]

    importancia = np.abs(valores).mean(axis=0)
    tabela = pd.DataFrame(
        {
            "feature": x_amostra.columns,
            "variavel": [FEATURE_LABELS.get(c, c) for c in x_amostra.columns],
            "mean_abs_shap": importancia,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    tabela.to_csv(
        destino / "shap_importancia.csv",
        index=False,
        encoding="utf-8-sig",
    )

    principais = tabela.head(12).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    barras = ax.barh(
        principais["variavel"],
        principais["mean_abs_shap"],
        color=COR_DESTAQUE,
    )
    ax.bar_label(
        barras,
        labels=[_decimal_pt(v) for v in principais["mean_abs_shap"]],
        padding=4,
        fontsize=8,
    )
    ax.set_title("Importância global das variáveis pelo SHAP")
    ax.set_xlabel("Média do impacto absoluto na saída do modelo")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(destino / "shap_importancia_global.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    nomes = [FEATURE_LABELS.get(c, c) for c in x_amostra.columns]
    shap.summary_plot(
        valores,
        x_amostra.to_numpy(),
        feature_names=nomes,
        max_display=12,
        color_bar_label="Valor da variável",
        plot_size=(9.2, 6.4),
        show=False,
    )
    figura = plt.gcf()
    eixo = plt.gca()
    eixo.set_title(
        "Resumo SHAP: direção e intensidade do impacto",
        fontsize=14,
    )
    eixo.set_xlabel(
        "Valor SHAP (impacto na saída do modelo)",
        fontsize=13,
        fontweight="bold",
    )
    eixo.tick_params(axis="x", labelsize=13, length=6, width=1.2)
    eixo.tick_params(axis="y", labelsize=11)

    if len(figura.axes) > 1:
        barra_cor = figura.axes[-1]
        barra_cor.set_ylabel("Valor da variável")
        barra_cor.set_yticks([0, 1])
        barra_cor.set_yticklabels(["Baixo", "Alto"])

    figura.tight_layout()
    figura.savefig(destino / "shap_resumo.png", dpi=300, bbox_inches="tight")
    plt.close(figura)

    return tabela
