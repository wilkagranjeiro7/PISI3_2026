# Análise Preditiva de Recuperação Fisiológica e Overstrain

Projeto acadêmico de Data Analytics, clustering e machine learning aplicado a dados de dispositivos vestíveis. O dashboard explora indicadores de sono, recuperação, frequência cardíaca e carga de treino, além de oferecer segmentação de usuários e uma classificação experimental de risco de overstrain.

> **Aviso:** os resultados e recomendações deste projeto têm finalidade exclusivamente acadêmica e não substituem avaliação médica ou acompanhamento profissional.

## Funcionalidades

- carregamento e limpeza do dataset;
- detecção e relatório de valores ausentes e outliers;
- análise exploratória e visualizações interativas;
- filtros e agrupamentos por características demográficas e esportivas;
- matriz de correlação e análises de Pareto;
- segmentação com MiniBatch K-Means, PCA, cotovelo e silhueta;
- comparação de modelos de classificação;
- métricas como acurácia, F1, MCC, AUC-ROC e matriz de confusão;
- explicabilidade opcional com SHAP;
- persistência do melhor modelo para a página de Insights;
- benchmark real de tamanho e leitura entre CSV e Parquet.

## Dataset

O projeto utiliza o [WHOOP Fitness Dataset](https://www.kaggle.com/datasets/likithagedipudi/whoop-fitness-dataset/data), com aproximadamente 100 mil registros.

O dataset não é versionado no Git por tamanho e licenciamento. Depois de obtê-lo na fonte original, use uma destas localizações:

- `Dashboard/whoop_fitness_dataset_100k.xlsx`;
- `whoop_fitness_dataset_100k.xlsx` na raiz;
- `Dashboard/data/dataset.xlsx`;
- `Dashboard/data/dataset.parquet`.

Na primeira execução, o sistema cria `Dashboard/data/dataset.pkl` como cache local. Dados, cache e modelos treinados são ignorados pelo Git.

## Tecnologias

- Python 3.10 a 3.12;
- Dash e Dash Bootstrap Components;
- Pandas, NumPy e SciPy;
- Plotly;
- Scikit-learn;
- OpenPyXL e PyArrow;
- Joblib.

XGBoost, LightGBM, CatBoost e SHAP são opcionais. Quando instalados, são detectados automaticamente.

## Estrutura

```text
PISI3_2026/
├── app.py                 # Ponto de entrada
├── model_manager.py       # Persistência dos modelos
├── requirements.txt       # Dependências
├── Dashboard/
│   ├── app.py             # Aplicação e rotas
│   ├── data_loader.py     # Carga, limpeza e features
│   ├── data/README.md     # Orientação para o dataset
│   └── pages/             # Páginas analíticas
└── Analises/              # Gráficos exportados
```

## Instalação no Windows

No PowerShell, a partir da raiz:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Adicione o dataset em uma das localizações documentadas e inicie:

```powershell
python app.py
```

Acesse [http://localhost:8050](http://localhost:8050).

## Preparação dos dados

O `DataManager`:

1. converte datas e valores numéricos;
2. valida faixas fisiologicamente plausíveis;
3. registra outliers substituídos por valores ausentes;
4. remove duplicidades;
5. cria as features derivadas;
6. calcula estatísticas descritivas;
7. salva um cache local para as próximas execuções.

Features derivadas:

- `sleep_quality = sleep_hours * (sleep_efficiency / 100)`;
- `strain_per_sleep = day_strain / (sleep_hours + 0.1)`;
- `hrv_ratio = hrv / (hrv_baseline + 1)`;
- `rhr_ratio = resting_heart_rate / (rhr_baseline + 1)`;
- `hrv_rhr_ratio = hrv / (resting_heart_rate + 1)`.

## Classificação de overstrain

O alvo acadêmico representa risco no mesmo dia:

```python
overstrain = (day_strain > mediana(day_strain)) & (hrv < hrv_baseline)
```

Para evitar vazamento de dados, o modelo bloqueia como entradas `day_strain`, `hrv`, `hrv_baseline` e as features derivadas diretamente dessas variáveis. A interface utiliza frequência cardíaca em repouso e indicadores de sono.

Modelos principais:

- Random Forest;
- Gradient Boosting;
- Regressão Logística;
- Árvore de Decisão;
- KNN;
- AdaBoost.

O conjunto é dividido aleatoriamente em 80% para treino e 20% para teste, com estratificação e semente fixa. Essa avaliação é demonstrativa: por se tratar de dados longitudinais, trabalhos futuros devem preferir validação temporal ou separação por usuário.

## Reprodutibilidade

- O dataset bruto, o cache e `saved_models/` não são enviados ao GitHub.
- Para refazer todo o processamento, exclua `Dashboard/data/dataset.pkl` e reinicie.
- Para usar a página de Insights após mudanças no pipeline, treine novamente um modelo na página Classificação.
- O benchmark de CSV e Parquet é medido no ambiente local; os resultados podem variar entre máquinas.

## Equipe

- Carlos Jonathan de Lima Malta
- Kassiane Gomes da Silva
- Leandro Augusto Barboza da Silva
- Leonardo Cassio da Silva Braz
- Wilka Vitória Granjeiro do Nascimento

## Contexto acadêmico

Desenvolvido para a disciplina de Projeto Interdisciplinar para Sistemas de Informação III — PISI3, 2026.
