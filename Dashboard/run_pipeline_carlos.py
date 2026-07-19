from __future__ import annotations

import json
from pathlib import Path

import joblib

from carlos_pipeline.config import (
    DEFAULT_OUTPUT_DIR,
    FEATURES,
    TARGET_COLUMN,
    TARGET_NAME,
    TARGET_THRESHOLD,
)
from carlos_pipeline.data import carregar_dados, preparar_target_e_features
from carlos_pipeline.evaluation import (
    avaliar_modelo,
    grafico_balanceamento,
    grafico_comparacao_3va,
    grafico_matriz_confusao,
    grafico_roc,
    salvar_metricas,
)
from carlos_pipeline.model import treinar_modelo
from carlos_pipeline.preprocessing import separar_preparar_e_balancear
from carlos_pipeline.xai import gerar_xai


def executar(caminho_dados: str | Path | None = None) -> dict:
    destino = DEFAULT_OUTPUT_DIR
    destino.mkdir(parents=True, exist_ok=True)

    dados_brutos = carregar_dados(caminho_dados)
    x, y = preparar_target_e_features(dados_brutos)
    preparados = separar_preparar_e_balancear(x, y)
    modelo = treinar_modelo(preparados)
    metricas, _, probabilidades = avaliar_modelo(modelo, preparados)

    metricas.update(
        {
            "target_column": TARGET_COLUMN,
            "target_name": TARGET_NAME,
            "target_rule": f"{TARGET_COLUMN} > {TARGET_THRESHOLD:g}",
            "features": FEATURES,
            "records_total": int(len(y)),
            "records_train_before_smote": int(
                sum(preparados.distribuicao_treino_antes.values())
            ),
            "records_train_after_smote": int(
                sum(preparados.distribuicao_treino_depois.values())
            ),
            "records_test": int(len(preparados.y_teste)),
            "train_distribution_before_smote": preparados.distribuicao_treino_antes,
            "train_distribution_after_smote": preparados.distribuicao_treino_depois,
            "test_distribution": {
                int(k): int(v)
                for k, v in preparados.y_teste.value_counts().sort_index().items()
            },
        }
    )

    grafico_balanceamento(preparados, destino)
    grafico_matriz_confusao(metricas, destino)
    grafico_roc(preparados.y_teste, probabilidades, metricas, destino)
    grafico_comparacao_3va(metricas, destino)
    gerar_xai(modelo, preparados.x_teste, destino)
    salvar_metricas(metricas, destino)

    joblib.dump(
        {
            "model": modelo,
            "imputer": preparados.imputer,
            "scaler": preparados.scaler,
            "features": FEATURES,
            "target_rule": metricas["target_rule"],
        },
        destino / "modelo_carlos.joblib",
    )

    with (destino / "LEIA-ME.txt").open("w", encoding="utf-8") as arquivo:
        arquivo.write(
            "Resultados gerados por Dashboard/run_pipeline_carlos.py.\n"
            "O SMOTE foi aplicado somente ao treino. O teste permaneceu intocado.\n"
            f"Acurácia final: {metricas['accuracy']:.4f}.\n"
            "Os 65,8% da 3VA são uma referência histórica; os protocolos diferem.\n"
            "Diferença para a referência da 3VA: "
            f"{metricas['accuracy_difference_pp']:.2f} pontos percentuais.\n"
        )

    print(json.dumps(metricas, ensure_ascii=False, indent=2))
    print(f"\nArquivos salvos em: {destino}")
    return metricas


if __name__ == "__main__":
    executar()
