import joblib
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# 1. Carregar os dados salvos
dados = joblib.load('lts.pkl')

# 2. Preparar os dados do SHAP
shap_df = pd.DataFrame(dados['shap_summary'])

# --- A MÁGICA DAS CORES AQUI ---
# Normalizamos os valores para criar a escala de cores
norm = mcolors.Normalize(vmin=shap_df['Importance'].min(), vmax=shap_df['Importance'].max())
# Escolhi a paleta 'coolwarm' (Azul para menores, Vermelho para maiores)
# Se quiser outras, pode trocar 'coolwarm' por 'viridis' ou 'plasma'
cores = cm.coolwarm(norm(shap_df['Importance'])) 

# 3. Desenhar o Gráfico
plt.figure(figsize=(10, 6))
# Passamos a nossa lista de cores variáveis no lugar do 'skyblue'
plt.barh(shap_df['Feature'], shap_df['Importance'], color=cores)

plt.xlabel('Importância (SHAP Value)')
plt.title('Quais variáveis mais impactam o Recovery?')
plt.gca().invert_yaxis() # Deixa a mais importante no topo
plt.tight_layout()
plt.show()

# 4. Mostrar as Métricas no terminal
print("--- Métricas do Modelo ---")
for chave, valor in dados['metrics'].items():
    if chave != 'conf_matrix': 
        print(f"{chave}: {valor:.3f}")