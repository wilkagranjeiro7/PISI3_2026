# Análise Preditiva de Recuperação Fisiológica e Overstrain com Machine Learning

Repositório voltado ao desenvolvimento de um ecossistema de Data Analytics, Clustering e Machine Learning para detecção de sobrecarga física (*overstrain*) utilizando dados de sensores vestíveis (*wearables*). Projeto desenvolvido para a disciplina de **Projeto Interdisciplinar para Sistemas de Informação III (PISI3 - 2026)**.

## Sobre o Projeto

O objetivo deste projeto é processar, analisar e modelar dados fisiológicos longitudinais para identificar padrões de fadiga crônica e prever o risco de **overstrain** — um estado de saturação biológica que precede lesões e queda severa de performance no esporte. 

A arquitetura do sistema foi projetada em módulos analíticos utilizando **Dash/Plotly**, contemplando:
1. **Módulo de Data Exploration & Profiling:** Diagnóstico estatístico automatizado e análise descritiva da integridade da base de dados.
2. **Módulo de Agrupamento (Clustering):** Algoritmos não-supervisionados para segmentação de perfis comportamentais e físicos de atletas.
3. **Pipeline de Classificação Supervisionada:** Ambiente sandbox para engenharia de features, tratamento de vazamento de dados, treinamento e validação cruzada de múltiplos modelos preditivos.
4. **Simulador de Insights & Heurísticas:** Interface interativa para testar a resposta do modelo treinado a partir de novos vetores de entrada de variáveis vitais.

## Conjunto de Dados (Dataset)

A base de estudo é o [WHOOP Fitness Dataset](https://www.kaggle.com/datasets/likithagedipudi/whoop-fitness-dataset/data) (via Kaggle).
* **Volume:** ~100.000 registros diários estruturados.
* **Amostragem:** Dados longitudinais de 286 usuários únicos anonimizados.
* **Métricas Fisiológicas Core:** Variabilidade da Frequência Cardíaca (VFC/HRV), Frequência Cardíaca em Repouso (RHR), Carga de Treino acumulada (`day_strain`), Horas Totais de Sono e Eficiência do Sono.

## 🗂️ Estrutura do projeto

```
PISI3_2026/
├── app.py                      # Ponto de entrada e inicialização da aplicação Dash
├── data_loader.py              # Singleton/Manager de carga, limpeza e engenharia de features
├── model_manager.py            # Orquestrador de serialização (salvamento/carga) de modelos preditivos
├── fitness_dashboard.txt       # Backlog de especificações e anotações técnicas do dashboard
├── requirements.txt            # Dependências e bibliotecas do ecossistema Python
├── whoop_fitness_dataset_100k.xlsx # Arquivo bruto/fonte de dados original
│
├── Dashboard/                  # Módulos core de visualização de dados
│   └── data/
│       └── dataset.parquet     # Dataset persistido em formato colunar de alta performance
│
├── pages/                      # Arquitetura de páginas e views multipáginas do Dash
│   ├── agrupamentos.py        # View para algoritmos não-supervisionados (Clustering)
│   ├── classificacao.py       # View para treino, pipeline e métricas supervisionadas
│   ├── dataframes.py          # View para visualização tabular e auditoria dos dados
│   ├── eda.py                 # View de Análise Exploratória de Dados
│   ├── filtros.py             # Componentes modulares de filtragem global de dados
│   ├── home.py                # Dashboard principal e Hub de navegação
│   ├── insights.py            # Motor de tomada de decisão e simulação de prontidão
│   ├── kmeans.py              # Implementações lógicas do agrupamento K-Means
│   ├── parquet.py             # Utilitários para conversão e otimização em formato Parquet
│   ├── plots.py               # Fábrica de gráficos, layouts e estilizações Plotly
│   └── profiling.py           # Relatórios de estatística descritiva e integridade dos dados
│
└── saved_models/               # Modelos binários serializados (.pkl) gerados pelo model_manager
```

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
