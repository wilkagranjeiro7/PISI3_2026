# CÉLULA: Figura 30 - Curva ROC Comparativa Cientificamente Rigorosa (XGBoost vs. Baseline Real)
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="whitegrid")

print("Gerando as curvas ROC para comparação...")

if 'modelo_final' in locals():
    modelo_principal = modelo_final
elif 'modelo_xgb' in locals():
    modelo_principal = modelo_xgb
else:
    from xgboost import XGBClassifier
    modelo_principal = XGBClassifier(random_state=42)
    modelo_principal.fit(X_train_balanced, y_train_balanced)

y_probs_xgb = modelo_principal.predict_proba(X_test_scaled)[:, 1]
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_probs_xgb)
auc_xgb = roc_auc_score(y_test, y_probs_xgb)


modelo_baseline = LogisticRegression(random_state=42)
modelo_baseline.fit(X_train_balanced, y_train_balanced)
y_probs_baseline = modelo_baseline.predict_proba(X_test_scaled)[:, 1]
fpr_base, tpr_base, _ = roc_curve(y_test, y_probs_baseline)
auc_base = roc_auc_score(y_test, y_probs_baseline)

plt.figure(figsize=(9, 7))

plt.plot(
    fpr_xgb, tpr_xgb, 
    label=f'Nosso Modelo Final (XGBoost) — AUC = {auc_xgb:.3f}', 
    color='#2ecc71', 
    linewidth=3.5
)


plt.plot(
    fpr_base, tpr_base, 
    label=f'Modelo Baseline (Regressão Linear) — AUC = {auc_base:.3f}', 
    color='#e74c3c', 
    linestyle='--', 
    linewidth=2.5
)


plt.plot([0, 1], [0, 1], color='#7f8c8d', linestyle=':', linewidth=2, label='Linha do Acaso (AUC = 0.500)')

plt.title('Figura 30: Curva ROC Comparativa — XGBoost vs. Baseline', fontsize=15, fontweight='bold', pad=15)
plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)', fontsize=12, fontweight='bold')
plt.ylabel('Taxa de Verdadeiros Positivos (Sensibilidade / Recall)', fontsize=12, fontweight='bold')
plt.legend(loc='lower right', fontsize=11, frameon=True)
plt.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()
print("Curva ROC rigorosa gerada com sucesso!")