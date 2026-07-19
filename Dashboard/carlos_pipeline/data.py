from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from .config import (
    DEFAULT_CACHE_PATH,
    DEFAULT_EXCEL_PATH,
    FEATURES,
    TARGET_COLUMN,
    TARGET_NAME,
    TARGET_THRESHOLD,
)


def carregar_dados(caminho: str | Path | None = None) -> pd.DataFrame:
    """Carrega primeiro o cache do dashboard e usa o Excel como alternativa."""
    if caminho is not None:
        origem = Path(caminho)
    elif DEFAULT_CACHE_PATH.exists():
        origem = DEFAULT_CACHE_PATH
    else:
        origem = DEFAULT_EXCEL_PATH

    if not origem.exists():
        raise FileNotFoundError(f"Base de dados não encontrada: {origem}")

    if origem.suffix.lower() == ".pkl":
        with origem.open("rb") as arquivo:
            conteudo = pickle.load(arquivo)
        dados = conteudo.get("df_clean") if isinstance(conteudo, dict) else conteudo
    elif origem.suffix.lower() in {".xlsx", ".xls"}:
        dados = pd.read_excel(origem, engine="openpyxl")
    else:
        raise ValueError("Use o cache .pkl ou a base original .xlsx.")

    if not isinstance(dados, pd.DataFrame):
        raise TypeError("O arquivo carregado não contém um DataFrame.")

    return dados.copy()


def preparar_target_e_features(dados: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    colunas_necessarias = [*FEATURES, TARGET_COLUMN]
    ausentes = [coluna for coluna in colunas_necessarias if coluna not in dados.columns]

    if ausentes:
        raise ValueError(f"Colunas ausentes na base: {', '.join(ausentes)}")

    trabalho = dados[colunas_necessarias].copy()

    for coluna in colunas_necessarias:
        trabalho[coluna] = pd.to_numeric(trabalho[coluna], errors="coerce")

    trabalho.loc[~trabalho["sleep_hours"].between(0, 24), "sleep_hours"] = pd.NA
    trabalho.loc[~trabalho["hrv"].between(10, 500), "hrv"] = pd.NA
    trabalho.loc[
        ~trabalho[TARGET_COLUMN].between(0, 100), TARGET_COLUMN
    ] = pd.NA
    trabalho = trabalho.dropna(subset=[TARGET_COLUMN])

    x = trabalho[FEATURES]
    y = (trabalho[TARGET_COLUMN] > TARGET_THRESHOLD).astype(int)
    y.name = TARGET_NAME

    if y.nunique() != 2:
        raise ValueError("O target precisa conter as duas classes.")

    return x, y
