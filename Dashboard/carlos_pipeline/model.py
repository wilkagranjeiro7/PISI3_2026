from sklearn.ensemble import HistGradientBoostingClassifier

from .config import RANDOM_STATE
from .preprocessing import DadosPreparados


def criar_modelo() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.06,
        max_leaf_nodes=25,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )


def treinar_modelo(dados: DadosPreparados) -> HistGradientBoostingClassifier:
    modelo = criar_modelo()
    modelo.fit(dados.x_treino_balanceado, dados.y_treino_balanceado)
    return modelo
