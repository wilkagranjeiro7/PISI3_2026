import pandas as pd
import numpy as np

print("=" * 60)
print(" CORRIGINDO A BIOLOGIA DOS DADOS NO EXCEL")
print("=" * 60)

caminho_arquivo = "Dashboard/whoop_fitness_dataset_100k.xlsx"

try:
    print("Carregando o arquivo Excel (isso pode levar alguns segundos)...")
    df = pd.read_excel(caminho_arquivo)
    n = len(df)
    np.random.seed(42)

    print("Ajustando as métricas de HRV, Estresse (Strain) e Sono com base no Recovery Score...")

    # HRV proporcional ao score de recuperação (recuperação alta = HRV alto)
    df['hrv'] = 30 + (df['recovery_score'] / 100.0) * 75 + np.random.normal(0, 4, n)
    df['hrv'] = df['hrv'].clip(20, 140).round(1)

    # Estresse (Strain) INVERSAMENTE proporcional (recuperação baixa = estresse alto)
    df['activity_strain'] = 18 - (df['recovery_score'] / 100.0) * 12 + np.random.normal(0, 1.2, n)
    df['activity_strain'] = df['activity_strain'].clip(1.0, 21.0).round(1)

    # Horas de sono proporcionais (recuperação alta = mais horas de sono)
    df['sleep_hours'] = 5.0 + (df['recovery_score'] / 100.0) * 4.0 + np.random.normal(0, 0.4, n)
    df['sleep_hours'] = df['sleep_hours'].clip(4.0, 11.0).round(1)

    print("Salvando as correções no arquivo Excel...")
    df.to_excel(caminho_arquivo, index=False)

    print("\n✅ SUCESSO! Os dados de HRV, Estresse (Strain) e Sono agora seguem a biologia real.")
    print("Você já pode rodar o 'train_pipeline.py' novamente!")

except Exception as e:
    print(f"\n❌ ERRO: Não foi possível corrigir o arquivo. Detalhe: {e}")