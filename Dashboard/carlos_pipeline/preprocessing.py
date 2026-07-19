from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE, TEST_SIZE


@dataclass
class DadosPreparados:
    x_treino_balanceado: pd.DataFrame
    y_treino_balanceado: pd.Series
    x_teste: pd.DataFrame
    y_teste: pd.Series
    imputer: SimpleImputer
    scaler: StandardScaler
    distribuicao_treino_antes: dict[int, int]
    distribuicao_treino_depois: dict[int, int]


def _distribuicao(y: pd.Series) -> dict[int, int]:
    contagem = y.value_counts().sort_index()
    return {int(classe): int(total) for classe, total in contagem.items()}


def separar_preparar_e_balancear(x: pd.DataFrame, y: pd.Series) -> DadosPreparados:
    """Separa antes do SMOTE para manter o conjunto de teste real e intocado."""
    x_treino, x_teste, y_treino, y_teste = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    x_treino_imputado = imputer.fit_transform(x_treino)
    x_teste_imputado = imputer.transform(x_teste)

    x_treino_padronizado = scaler.fit_transform(x_treino_imputado)
    x_teste_padronizado = scaler.transform(x_teste_imputado)

    x_treino_df = pd.DataFrame(
        x_treino_padronizado,
        columns=x.columns,
        index=x_treino.index,
    )
    x_teste_df = pd.DataFrame(
        x_teste_padronizado,
        columns=x.columns,
        index=x_teste.index,
    )

    distribuicao_antes = _distribuicao(y_treino)
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    x_balanceado, y_balanceado = smote.fit_resample(x_treino_df, y_treino)

    x_balanceado = pd.DataFrame(x_balanceado, columns=x.columns)
    y_balanceado = pd.Series(y_balanceado, name=y.name)

    return DadosPreparados(
        x_treino_balanceado=x_balanceado,
        y_treino_balanceado=y_balanceado,
        x_teste=x_teste_df,
        y_teste=y_teste.reset_index(drop=True),
        imputer=imputer,
        scaler=scaler,
        distribuicao_treino_antes=distribuicao_antes,
        distribuicao_treino_depois=_distribuicao(y_balanceado),
    )
