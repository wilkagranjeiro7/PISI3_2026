# 🚀 FitMatch: Análise Preditiva de Recuperação Fisiológica e Machine Learning

> Repositório oficial do ecossistema de Data Analytics, Balanceamento de Dados (SMOTE) e Machine Learning voltado à previsão de recuperação (*recovery*) de atletas utilizando dados de sensores vestíveis (*wearables*). Projeto desenvolvido para a disciplina de **Projeto Interdisciplinar para Sistemas de Informação III (PISI3 - UFRPE, 2026)**.

---

## 📌 Sobre o Projeto

O **FitMatch** é uma aplicação inteligente focada na recomendação de exercícios físicos e monitoramento de performance esportiva. O principal objetivo é processar, analisar e modelar dados fisiológicos longitudinais para classificar o nível de recuperação diária (*recovery*: Baixa, Moderada, Alta), prevenindo quadros de fadiga crônica e otimizando a prescrição de treinos.

A arquitetura do sistema foi estruturada em módulos analíticos utilizando **Dash/Plotly**, contemplando:
1. **Módulo de Classificação Avançada:** Pipeline preditivo baseado em Random Forest, aliado ao SMOTE para tratamento de desbalanceamento de classes e acompanhamento de métricas (Curva ROC e Matriz de Confusão).
2. **Explicabilidade (XAI - SHAP):** Análise de impacto global das variáveis preditivas (como calorias gastas, HRV e horas de sono) com visual customizado em degradê azul e integração total ao tema escuro.
3. **Módulo de Visualização de Dados e EDA:** Painéis interativos para exploração descritiva, auditoria e tratamento do dataset original.

---

## 📊 Conjunto de Dados (Dataset)

A base utilizada é o [WHOOP Fitness Dataset](https://www.kaggle.com/datasets/likithagedipudi/whoop-fitness-dataset/data) obtido via Kaggle.
* **Volume:** ~100.000 registros diários estruturados.
* **Métricas Fisiológicas Core:** Variabilidade da Frequência Cardíaca (VFC/HRV), Frequência Cardíaca em Repouso (RHR), Carga de Treino acumulada (`calories_burned`, `activity_strain`), Horas Totais de Sono e Desempenho do Sono (`sleep_hours`, `sleep_performance`).

---

## 🗂️ Estrutura do Projeto

```text
PISI3_2026/
├── train_pipeline.py          # Script de treinamento, balanceamento (SMOTE) e geração de artefatos
├── lts.pkl                    # Arquivo binário com métricas e distribuições para o Dashboard
├── requirements.txt           # Dependências e bibliotecas do ecossistema Python
│
├── Dashboard/                 # Módulos core de visualização e interface Dash
│   ├── app.py                 # Ponto de entrada e inicialização do servidor web
│   ├── assets/                # Arquivos estáticos (ex: gráficos SHAP exportados)
│   │   └── shap_bar_plot.png  # Gráfico XAI estilizado com degradê azul
│   └── pages/                 # Arquitetura de páginas multipáginas do Dash
│       ├── advanced_classification.py # Página principal de métricas, SMOTE, ROC e XAI
│       ├── eda.py             # Análise Exploratória de Dados
│       ├── home.py            # Hub de navegação principal
│       └── ...                # Demais páginas e filtros analíticos
│
└── saved_models/              # Modelos binários serializados (.pkl)

##  Como Executar o Ambiente

1. **Clone o repositório:**
```bash
   git clone [git@github.com:wilkagranjeiro7/PISI3_2026.git]
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


## 🧠 Pipeline de Machine Learning
O pipeline matemático de processamento adiciona dinamicamente as seguintes variáveis derivadas ao conjunto de dados:

### Features derivadas
A partir das colunas brutas do dataset, são calculadas automaticamente:
- `Qualidade do Sono(sleep_quality)` = `sleep_hours * (sleep_efficiency / 100)`
- `Strain por Hora de Sono(strain_per_sleep)` = `day_strain / (sleep_hours + 0.1)`
- `Razão VFC(hrv_ratio)` = `hrv / (hrv_baseline + 1)`
- `Razão RHR(rhr_ratio)` = `resting_heart_rate / (rhr_baseline + 1)`
- `Proporção Cardio-Fisiológica(hrv_rhr_ratio)` = `hrv / (resting_heart_rate + 1)`

### Rótulo (target) de Overstrain
```python
overstrain = (day_strain > mediana(day_strain)) AND (hrv < hrv_baseline)
```

⚠️ **Atenção a vazamento de dados (data leakage):** como o rótulo é construído a partir de `day_strain`, `hrv` e `hrv_baseline`, essas colunas — e qualquer feature derivada delas (`hrv_ratio`, `hrv_rhr_ratio`, `strain_per_sleep`) — são **bloqueadas como features de entrada do modelo**. Usá-las inflaria artificialmente a acurácia (chegamos a observar 100% de acurácia/F1 antes da correção, justamente por esse motivo).

A página de Classificação permite escolher entre dois modos de alvo temporal:
- **Dia seguinte** (recomendado): prevê se haverá overstrain amanhã, com base nos dados de hoje — evita qualquer ambiguidade de causalidade.
- **Mesmo dia**: prevê o overstrain do dia atual, usando apenas features que não compõem o rótulo.

### Modelos disponíveis
Random Forest, Gradient Boosting, Regressão Logística, Árvore de Decisão, KNN, AdaBoost (e XGBoost/LightGBM/CatBoost, se instalados).

O melhor modelo é selecionado por um score ponderado (AUC, F1, MCC, acurácia) e salvo automaticamente em `saved_models/` para uso na página de Insights.

## 🔍 Achados metodológicos

Durante o desenvolvimento, identificamos que a definição original do rótulo de overstrain causava **vazamento de dados** quando `day_strain`/`hrv` eram usados como features, resultando em métricas artificialmente perfeitas (100% de acurácia). Após a correção (bloqueio dessas colunas + avaliação com alvo no dia seguinte), as métricas caíram para próximo do nível aleatório (AUC ≈ 0.50), indicando que **as features de sono e frequência cardíaca em repouso isoladamente não têm poder preditivo forte sobre o overstrain do dia seguinte** neste dataset.

Esse resultado é documentado como parte da análise crítica do projeto, e motivou a exploração de abordagens alternativas (ex.: alvo no mesmo dia, agregações de múltiplos dias).

## 👥 Equipe

_Carlos Jonathan de Lima Malta_
_Kassiane Gomes da Silva_
_Leandro Augusto Barboza da Silva_
_Leonardo Cassio da Silva Braz_
_Wilka Vitória Granjeiro do Nascimento_

## 📄 Licença

Projeto acadêmico desenvolvido para a disciplina de Projeto Interdisciplinar para Sistemas de Informação III
