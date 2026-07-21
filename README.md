# FitMatch: Dashboard de Wearables e Pipeline de Classificação Avançada

Repositório voltado ao desenvolvimento de um ecossistema de Data Analytics, Clustering e Machine Learning para análise de dados de sensores vestíveis (*wearables*). Projeto desenvolvido para a disciplina de **Projeto Interdisciplinar para Sistemas de Informação III (PISI3 - 2026)**.

## Sobre o Projeto

O **FitMatch** é uma aplicação web interativa desenvolvida em **Dash/Plotly** que integra um pipeline completo de ciência de dados e aprendizado de máquina para analisar métricas fisiológicas e classificar estados de recuperação e desempenho de atletas.

A arquitetura do sistema contempla:

1. **Módulo de Data Exploration & Profiling:** Diagnóstico estatístico e análise descritiva da base de dados.
2. **Módulo de Agrupamento (Clustering):** Algoritmos não-supervisionados para segmentação de perfis físicos.
3. **Pipeline de Classificação Avançada:** Ambiente integrado com balanceamento de dados (**SMOTE**), modelos probabilísticos (**Naive Bayes**) e gradient boosting (**LightGBM - 3VA**), acompanhado de Matriz de Confusão, Curva ROC Multiclasse e Explicabilidade (**SHAP**).
4. **Hub de Navegação e Visualizações:** Gráficos interativos de comparação de modelos e impacto de variáveis.

## Conjunto de Dados (Dataset)

A base de estudo é o **WHOOP Fitness Dataset** (via Kaggle).

* **Volume:** ~100.000 registros diários estruturados.
* **Métricas Fisiológicas Principais:** Variabilidade da Frequência Cardíaca (VFC/HRV), Frequência Cardíaca em Repouso (`resting_heart_rate`), Carga de Atividade (`activity_strain`), Horas e Desempenho de Sono (`sleep_hours`, `sleep_performance`) e Calorias Queimadas (`calories_burned`).

## 🗂️ Estrutura do Projeto

```
PISI3_2026/
├── app.py                                   # Ponto de entrada e inicialização da aplicação Dash
├── data_loader.py                           # Singleton/Manager de carga e limpeza de dados
├── train_pipeline.py                        # Script de treinamento do pipeline e serialização (lts.pkl)
├── lts.pkl                                  # Objeto serializado contendo métricas, modelos e dados de teste
├── requirements.txt                         # Dependências e bibliotecas do ecossistema Python
├── whoop_fitness_dataset_100k.xlsx          # Arquivo bruto/fonte de dados original
├── corrigir_dados_kassiane.py               # Script utilitário para correção e tratamento de dados
├── visualizar_resultados_kassiane.py        # Script para auditoria e exibição de resultados
│
├── Dashboard/                               # Módulos core de visualização e rotas
│   ├── app.py                               # Configuração principal das rotas do Dashboard
│   ├── assets/                              # Estilos, ícones e arquivos estáticos
│   │   └── shap_bar_plot.png                # Imagem estática gerada do gráfico SHAP
│   ├── data/
│   │   └── dataset.parquet                  # Dataset otimizado em formato colunar
│   └── pages/                               # Páginas multipáginas do Dash
│       ├── agrupamentos.py                  # Visualização de clusters e agrupamentos
│       ├── classificacao_kassiane.py        # Pipeline de Naive Bayes, SMOTE, ROC e SHAP
│       ├── eda_kassiane.py                  # Análise Exploratória de Dados personalizada
│       ├── home.py                          # Hub principal de navegação e boas-vindas
│       └── profiling.py                     # Relatórios estatísticos descritivos
│
└── Resultados_Kassiane/                     # Evidências gráficas e capturas de tela dos resultados
    ├── analisecondicional.png
    ├── explicabilidade.png
    ├── graficocomparacao.png
    ├── matriz.png
    ├── roc.png
    └── smote.png
```

## 🧠 Pipeline de Machine Learning & Modelagem

O pipeline de classificação supervisionada processa as classes alvo divididas em 3 categorias de recuperação (`Baixa`, `Moderada`, `Alta`):

1. **Balanceamento de Classes (SMOTE):** Correção do desbalanceamento original das categorias no dataset para evitar viés preditivo.
2. **Modelos Principais:**
   - **Naive Bayes:** Modelo probabilístico base para classificação multiclasse, gerando as probabilidades (`y_proba`) e avaliações de Curva ROC.
   - **LightGBM (3VA):** Modelo de alta performance para comparação de acurácia e F1-Score.
3. **Avaliação e Explicabilidade:**
   - **Curva ROC Multiclasse** com cálculo de AUC por classe.
   - **Matriz de Confusão** interativa com heatmaps otimizados.
   - **Importância das Variáveis (SHAP / Explicabilidade):** Identificação do impacto absoluto e queda média na acurácia por variável fisiológica (`hrv`, `sleep_hours`, `activity_strain`, etc.).

## 🚀 Como Executar o Ambiente

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/wilkagranjeiro7/PISI3_2026.git
   cd PISI3_2026
   ```

2. **Configure e ative o ambiente virtual:**
   ```bash
   python -m venv venv
   # No Windows:
   venv\Scripts\activate
   # No Linux/Mac:
   source venv/bin/activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Inicie o servidor local do Dashboard:**
   ```bash
   python app.py
   ```
   Acesse `http://localhost:8050` no navegador.

## 👥 Equipe

_Carlos Jonathan de Lima Malta_
_Kassiane Gomes da Silva_
_Leandro Augusto Barboza da Silva_
_Leonardo Cassio da Silva Braz_
_Wilka Vitória Granjeiro do Nascimento_

## 📄 Licença

Projeto acadêmico desenvolvido para a disciplina de Projeto Interdisciplinar para Sistemas de Informação III
