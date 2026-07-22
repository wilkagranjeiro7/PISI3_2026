# Pipeline Preditivo de Prontidão Física 

Este repositório contém a documentação, os códigos e os resultados do pipeline analítico e de Machine Learning desenvolvido para prever a prontidão física e otimizar a prescrição de treinos com base em dados de dispositivos *wearables* (Whoop).

## Visão Geral do Projeto
O objetivo principal foi transformar dados brutos de saúde (sono, variabilidade cardíaca, esforço e perfil físico) em respostas práticas, automatizadas e seguras para o usuário final, atuando ativamente na prevenção de *overtraining* e lesões.

## Tecnologias e Metodologia
- **Ambiente de Desenvolvimento:** Google Colab
- **Linguagem:** Python
- **Manipulação e Análise:** Pandas, NumPy, Scikit-Learn
- **Balanceamento de Dados:** SMOTE (Synthetic Minority Over-sampling Technique) aplicado exclusivamente no conjunto de treino para evitar vazamento de dados (*data leakage*).
- **Modelagem Preditiva (Benchmark):** Teste de múltiplos algoritmos, consagrando o **XGBoost (Extreme Gradient Boosting)** como o motor oficial devido à sua superioridade em capturar não-linearidades biológicas.
- **Inteligência Artificial Explicável (XAI):** Aplicação de valores **SHAP** para abrir a "caixa-preta" da IA e validar o aprendizado com base na literatura de medicina esportiva.

## Principais Resultados
- **Acurácia Final:** Salto de 65,80% (baseline anterior 3VA) para **69,26% / 70%** com o XGBoost estruturado.
- **Poder Discriminatório (Curva ROC):** AUC de **0,752** (superando o baseline linear de 0,600).
- **Validação Biológica:** Comprovação através do SHAP de que a **VFC (HRV)** e o **Sono** são os pilares determinantes na tomada de decisão do modelo.

## Trabalhos Futuros
- Implementação de arquitetura em cascata (Two-Stage Model): um classificador para filtragem de risco de fadiga seguido de um regressor para cálculo da dosagem exata de esforço.
- Inclusão de variáveis contextuais diárias (hidratação, nutrição e estresse mental).