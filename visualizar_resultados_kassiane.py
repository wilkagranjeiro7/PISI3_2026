import joblib
import matplotlib.pyplot as plt
import pandas as pd

# ==============================================================================
# 1. CARREGAR DADOS E MÉTRICAS SALVAS
# ==============================================================================
print("=" * 60)
print(" CARREGANDO DADOS E MÉTRICAS DO MODELO")
print("=" * 60)

df = pd.read_excel("Dashboard/whoop_fitness_dataset_100k.xlsx")

# Recriar a categoria de recuperação (target)
df["recovery_category"] = pd.cut(
    df["recovery_score"],
    bins=[0, 33, 66, 100],
    labels=["Baixa", "Moderada", "Alta"]
)
df.dropna(subset=["recovery_category"], inplace=True)

# Carregar o arquivo lts.pkl
dados = joblib.load('lts.pkl')

print("\n--- Métricas do Naive Bayes ---")
for chave, valor in dados['metrics'].items():
    if isinstance(valor, (int, float)):
        print(f"• {chave}: {valor:.3f}")

print("\n--- Comparação com o LightGBM (3VA) ---")
for chave, valor in dados['metrics_lgbm'].items():
    print(f"• {chave}: {valor:.3f}")

# ==============================================================================
# 2. PLOTAR OS DOIS GRÁFICOS LADO A LADO
# ==============================================================================
cores_categorias = {'Baixa': '#E8968C', 'Moderada': '#D4A574', 'Alta': '#7CB3A1'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0D0D0D')

# --- GRÁFICO 1: HRV Médio por Categoria ---
df_hrv_mean = df.groupby('recovery_category', observed=False)['hrv'].mean().reset_index()
bars1 = ax1.bar(df_hrv_mean['recovery_category'], df_hrv_mean['hrv'], 
               color=[cores_categorias[cat] for cat in df_hrv_mean['recovery_category']], 
               edgecolor='#555555', width=0.6)

ax1.set_facecolor('#1A1A1A')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('#555555')
ax1.spines['bottom'].set_color('#555555')
ax1.tick_params(colors='white')
ax1.set_ylabel("HRV Médio (ms)", color='white', fontsize=11)
ax1.set_title("HRV Médio por Categoria", color='white', fontsize=13, pad=12)

for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}ms', ha='center', va='bottom', color='white', fontsize=10)

# --- GRÁFICO 2: Horas de Sono por Categoria ---
df_sono_mean = df.groupby('recovery_category', observed=False)['sleep_hours'].mean().reset_index()
bars2 = ax2.bar(df_sono_mean['recovery_category'], df_sono_mean['sleep_hours'], 
               color=[cores_categorias[cat] for cat in df_sono_mean['recovery_category']], 
               edgecolor='#555555', width=0.6)

ax2.set_facecolor('#1A1A1A')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#555555')
ax2.spines['bottom'].set_color('#555555')
ax2.tick_params(colors='white')
ax2.set_ylabel("Média de Horas de Sono", color='white', fontsize=11)
ax2.set_title("Média de Sono por Categoria", color='white', fontsize=13, pad=12)

for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f'{yval:.1f}h', ha='center', va='bottom', color='white', fontsize=10)

plt.tight_layout()
plt.show()