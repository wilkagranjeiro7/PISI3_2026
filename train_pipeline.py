import os
import sys
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.inspection import permutation_importance

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# ==============================================================================
# 1. CARREGAR DADOS
# ==============================================================================
print("=" * 70)
print("CARREGANDO DATASET...")
print("=" * 70)

df = pd.read_excel("Dashboard/whoop_fitness_dataset_100k.xlsx")

# ==============================================================================
# 2. DEFINIÇÃO DO TARGET
# ==============================================================================
df["recovery_category"] = pd.cut(
    df["recovery_score"],
    bins=[0, 33, 66, 100],
    labels=["Baixa", "Moderada", "Alta"]
)
df.dropna(subset=["recovery_category"], inplace=True)

# ==============================================================================
# 3. SELEÇÃO DAS FEATURES
# ==============================================================================
features = ["hrv", "sleep_hours", "resting_heart_rate", "activity_strain", 
            "calories_burned", "sleep_performance", "hrv_baseline"]

X = df[features]
y = df["recovery_category"]
dist_antes = y.value_counts().to_dict()

# ==============================================================================
# 4. DIVISÃO TREINO / TESTE
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# ==============================================================================
# 5. PIPELINE (SMOTE + NAIVE BAYES)
# ==============================================================================
print("\nTREINANDO NAIVE BAYES (COM SMOTE)...")
pipeline = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("classifier", GaussianNB())
])

pipeline.fit(X_train, y_train)

X_resampled, y_resampled = pipeline.named_steps["smote"].fit_resample(X_train, y_train)
dist_depois = pd.Series(y_resampled).value_counts().to_dict()

# ==============================================================================
# 6. AVALIAÇÃO DO MODELO ATUAL (NAIVE BAYES)
# ==============================================================================
y_pred = pipeline.predict(X_test)
metrics_nb = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
    "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
    "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    "conf_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "classes": pipeline.classes_.tolist()
}

# ==============================================================================
# 7. COMPARAÇÃO COM O MELHOR MODELO DA 3VA (LIGHTGBM)
# ==============================================================================
metrics_lgbm = {
    "accuracy": 0.952,
    "precision": 0.951,
    "recall": 0.952,
    "f1_score": 0.951
}

print("\n--- COMPARAÇÃO DE RESULTADOS ---")
print(f"LightGBM (3VA) - Acurácia: {metrics_lgbm['accuracy']:.3f} | F1: {metrics_lgbm['f1_score']:.3f}")
print(f"Naive Bayes   - Acurácia: {metrics_nb['accuracy']:.3f} | F1: {metrics_nb['f1_score']:.3f}")

# ==============================================================================
# 8. CÁLCULO DE IMPORTÂNCIA DAS VARIÁVEIS (Normalizado para Proporção 0-100%)
# ==============================================================================
print("\nCALCULANDO IMPORTÂNCIA DAS VARIÁVEIS...")
resultado_importancia = permutation_importance(pipeline, X_test, y_test, n_repeats=5, random_state=42)

raw_importances = resultado_importancia.importances_mean
importances_pos = np.clip(raw_importances, 0, None)

if importances_pos.sum() > 0:
    importances_relativas = (importances_pos / importances_pos.sum()) * 100
else:
    importances_relativas = importances_pos * 100

shap_summary = []
for i, feature in enumerate(features):
    shap_summary.append({
        "feature": feature,
        "importance": round(importances_relativas[i], 1)
    })

shap_summary = sorted(shap_summary, key=lambda x: x["importance"], reverse=True)

# ==============================================================================
# 9. GERAÇÃO DOS GRÁFICOS EM IMAGEM
# ==============================================================================
print("\nGERANDO OS GRÁFICOS...")
os.makedirs("Dashboard/assets", exist_ok=True)

cores_categorias = {'Baixa': '#2A4B6B', 'Moderada': '#528AB5', 'Alta': '#89C2EB'}

# --- GRÁFICO 1: Média de HRV ---
df_hrv_mean = df.groupby('recovery_category', observed=False)['hrv'].mean().reset_index()
fig, ax = plt.subplots(figsize=(8, 5), facecolor='none')
ax.set_facecolor('none')
bars1 = ax.bar(df_hrv_mean['recovery_category'], df_hrv_mean['hrv'], 
               color=[cores_categorias[cat] for cat in df_hrv_mean['recovery_category']], 
               edgecolor='#555555', width=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#555555')
ax.spines['bottom'].set_color('#555555')
ax.tick_params(colors='white')
ax.set_ylabel("HRV Médio (ms)", color='white', fontsize=11)
ax.set_title("HRV Médio por Categoria de Recuperação", color='white', fontsize=13, pad=12)
for bar in bars1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}ms', ha='center', va='bottom', color='white', fontsize=10)
plt.tight_layout()
plt.savefig("Dashboard/assets/grafico_hrv.png", dpi=300, transparent=True)
plt.close()

# --- GRÁFICO 2: Média de Sono ---
df_sono_mean = df.groupby('recovery_category', observed=False)['sleep_hours'].mean().reset_index()
fig, ax = plt.subplots(figsize=(8, 5), facecolor='none')
ax.set_facecolor('none')
bars2 = ax.bar(df_sono_mean['recovery_category'], df_sono_mean['sleep_hours'], 
               color=[cores_categorias[cat] for cat in df_sono_mean['recovery_category']], 
               edgecolor='#555555', width=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#555555')
ax.spines['bottom'].set_color('#555555')
ax.tick_params(colors='white')
ax.set_ylabel("Média de Horas de Sono", color='white', fontsize=11)
ax.set_title("Média de Horas de Sono por Categoria de Recuperação", color='white', fontsize=13, pad=12)
for bar in bars2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f'{yval:.1f}h', ha='center', va='bottom', color='white', fontsize=10)
plt.tight_layout()
plt.savefig("Dashboard/assets/grafico_sono.png", dpi=300, transparent=True)
plt.close()

# --- GRÁFICO 3: Impacto do Estresse (Strain) ---
print("Gerando Gráfico 3: Impacto do Estresse (Strain)...")
df_strain_mean = df.groupby('recovery_category', observed=False)['activity_strain'].mean().reset_index()
fig, ax = plt.subplots(figsize=(8, 5), facecolor='none')
ax.set_facecolor('none')
bars3 = ax.bar(df_strain_mean['recovery_category'], df_strain_mean['activity_strain'], 
               color=[cores_categorias[cat] for cat in df_strain_mean['recovery_category']], 
               edgecolor='#555555', width=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#555555')
ax.spines['bottom'].set_color('#555555')
ax.tick_params(colors='white')
ax.set_ylabel("Nível de Estresse / Strain Médio", color='white', fontsize=11)
ax.set_title("Impacto do Estresse Diário (Strain) na Recuperação", color='white', fontsize=13, pad=12)
for bar in bars3:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.1f}', ha='center', va='bottom', color='white', fontsize=10)
plt.tight_layout()
plt.savefig("Dashboard/assets/grafico_estresse_strain.png", dpi=300, transparent=True)
plt.close()

# ==============================================================================
# 10. SALVAMENTO DOS DADOS PARA O DASHBOARD (`lts.pkl`)
# ==============================================================================
print("\nSALVANDO DADOS PARA O DASHBOARD...")
y_proba = pipeline.predict_proba(X_test)
joblib.dump({
    "metrics": metrics_nb,
    "metrics_lgbm": metrics_lgbm,
    "dist_antes": dist_antes,
    "dist_depois": dist_depois,
    "shap_summary": shap_summary,
    "y_test": y_test,
    "y_proba": y_proba,
    "classes": pipeline.classes_
}, "lts.pkl")

print("PIPELINE CONCLUÍDO E TODOS OS ARQUIVOS SALVOS COM SUCESSO!")