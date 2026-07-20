import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import shap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

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
# 5. PIPELINE (SMOTE + RANDOM FOREST)
# ==============================================================================
print("\nTREINANDO RANDOM FOREST...")
pipeline = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])

pipeline.fit(X_train, y_train)

# Distribuição após SMOTE
X_resampled, y_resampled = pipeline.named_steps["smote"].fit_resample(X_train, y_train)
dist_depois = pd.Series(y_resampled).value_counts().to_dict()

# ==============================================================================
# 6. AVALIAÇÃO
# ==============================================================================
y_pred = pipeline.predict(X_test)
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
    "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
    "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    "conf_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "classes": pipeline.classes_.tolist()
}

# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# 7. SHAP (XAI) - VISUAL IDÊNTICO À REFERÊNCIA (COM DEGRADÊ E COLORBAR)
# ==============================================================================
print("\nGERANDO GRÁFICO SHAP...")
import matplotlib.colors as mcolors

rf_model = pipeline.named_steps["classifier"]
explainer = shap.TreeExplainer(rf_model)

sample_size = min(300, len(X_test))
X_test_sample = X_test.sample(n=sample_size, random_state=42)
shap_values = explainer.shap_values(X_test_sample)

# Tratamento para garantir o tamanho correto
if isinstance(shap_values, list):
    vals = np.array(shap_values)
    mean_shap_values = np.abs(vals).mean(axis=1).mean(axis=0)
else:
    if hasattr(shap_values, "values"):
        vals = shap_values.values
    else:
        vals = np.array(shap_values)
        
    if len(vals.shape) == 3:
        mean_shap_values = np.abs(vals).mean(axis=0).mean(axis=1)
    else:
        mean_shap_values = np.abs(vals).mean(axis=0)

df_shap = pd.DataFrame({'feature': X_test_sample.columns, 'importance': mean_shap_values})

# TRUQUE VISUAL: Se a importância for 0, coloca um valor minúsculo só para desenhar a pontinha da barra
max_imp = df_shap['importance'].max()
tiny_value = max_imp * 0.015  # 1.5% do tamanho da maior barra
df_shap.loc[df_shap['importance'] == 0, 'importance'] = tiny_value

df_shap = df_shap.sort_values(by='importance', ascending=True)

# ---------------------------------------------------------
# CONFIGURAÇÃO DE ESTILO: DEGRADÊ E COLORBAR
# ---------------------------------------------------------
os.makedirs("Dashboard/assets", exist_ok=True)
fig, ax = plt.subplots(figsize=(10, 7), facecolor='none')
ax.set_facecolor('none')

# Criando o mapa de cores (Degradê Azul) baseado nos valores de importância
cmap = plt.cm.Blues
norm = mcolors.Normalize(vmin=df_shap['importance'].min(), vmax=df_shap['importance'].max())
cores = cmap(norm(df_shap['importance'].values))

# Desenhando as barras com as cores mapeadas e uma borda suave
barras = ax.barh(df_shap['feature'], df_shap['importance'], color=cores, edgecolor='#555555', linewidth=0.5)

# Estilizando os eixos para o modo escuro
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#555555')
ax.spines['bottom'].set_color('#555555')

ax.tick_params(colors='white')
ax.set_xlabel("Importance", color='white', fontsize=12)
ax.set_ylabel("Feature", color='white', fontsize=12)

# Adicionando a barra lateral de cores (Colorbar) igual à foto
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.ax.yaxis.set_tick_params(color='white')
cbar.ax.tick_params(labelsize=10, colors='white')
cbar.set_label('Importance', color='white', fontsize=12)

# A anotação amarela do sleep_performance foi completamente removida!

plt.title("Importância das Variáveis (SHAP)", fontsize=14, pad=15, color='white')
plt.tight_layout()

# Salvando com fundo totalmente transparente
plt.savefig("Dashboard/assets/shap_bar_plot.png", dpi=300, transparent=True)
plt.close()

print("Gráfico SHAP gerado com sucesso!")
# ==============================================================================
# 8. SALVAMENTO FINAL
# ==============================================================================
y_proba = pipeline.predict_proba(X_test)
joblib.dump({
    "metrics": metrics,
    "dist_antes": dist_antes,
    "dist_depois": dist_depois,
    "shap_summary": "Feito",
    "y_test": y_test,
    "y_proba": y_proba,
    "classes": pipeline.classes_
}, "lts.pkl")

print("\nPIPELINE CONCLUÍDO COM SUCESSO!")