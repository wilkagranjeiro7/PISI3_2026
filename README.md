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

## 🛠️ Implementações e Ajustes

* **Pipeline de Classificação Supervisionada:** Construção de um fluxo completo de Machine Learning em Python para prever estados de recuperação de atletas divididos em 3 categorias (`Baixa`, `Moderada`, `Alta`) usando o WHOOP Fitness Dataset.
* **Balanceamento de Dados (SMOTE):** Implementação da técnica SMOTE para corrigir o desbalanceamento natural das classes de recuperação na base de dados.
* **Modelagem Probabilística (Naive Bayes & LightGBM):** Configuração do Naive Bayes como modelo principal de classificação (para gerar distribuições de probabilidade precisas) e do LightGBM (3VA) como modelo de alta performance para comparação de acurácia e F1-Score.
* **Visualizações e Métricas Avançadas no Dash:**
   * **Curva ROC Multiclasse:** Alinhamento matemático das colunas de probabilidade do modelo com as classes reais, garantindo curvas e valores de AUC corretos e realistas em um layout compacto e quadrado.
   * **Matriz de Confusão Interativa:** Heatmap com tons de azul e contraste inteligente para facilitar a leitura dos acertos e erros do modelo.
   * **Gráfico de Explicabilidade (SHAP):** Personalização da exibição da importância das variáveis, tratando variáveis com impacto nulo ou mínimo (como `sleep_performance` e `hrv_baseline`) para exibirem explicitamente o valor `0` com bloquinhos proporcionais e limpos, mantendo a hierarquia correta entre as demais métricas (como `resting_heart_rate` e `calories_burned`).

## 📌 Justificativas Técnicas

**Por que usar o SMOTE?**
* *Como escolhi:* Identificando que o dataset de wearables costuma ter distribuições desiguais entre dias de alta e baixa recuperação.
* *Por quê:* Para evitar que o algoritmo ficasse viciado apenas na classe majoritária, garantindo um treinamento justo e equilibrado para todas as categorias.

**Por que escolher o Naive Bayes e o LightGBM?**
* *Como escolhi:* Testando abordagens probabilísticas e de Gradient Boosting focadas em eficiência.
* *Por quê:* O Naive Bayes foi escolhido por sua robustez estatística em classificar probabilidades (`predict_proba`), o que permitiu traçar a Curva ROC multiclasse com facilidade. Já o LightGBM entrou como uma alternativa moderna de árvore de decisão para contrapor as métricas de desempenho.

**Por que incluir Curva ROC, Matriz de Confusão e SHAP?**
* *Como escolhi:* Buscando uma avaliação de 360 graus do modelo preditivo.
* *Por quê:*
   * A Matriz de Confusão mostra onde o modelo está errando na prática.
   * A Curva ROC valida a capacidade discriminatória de cada classe separadamente.
   * O SHAP foi escolhido por ser o padrão ouro em explicabilidade de IA — fundamental em projetos voltados a saúde e wearables, pois explica exatamente quais fatores fisiológicos pesaram na decisão do modelo.

**Por que ajustar o visual dos gráficos (tamanhos de blocos e zeros)?**
* *Como escolhi:* Refinando a interface de usuário (UI) do Dashboard em Dash/Plotly.
* *Por quê:* Para garantir uma experiência de apresentação impecável, limpa e profissional, onde variáveis irrelevantes não poluem visualmente a tela e mantêm uma hierarquia lógica compreensível para quem está assistindo.

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
