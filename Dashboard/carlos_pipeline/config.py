from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET_COLUMN = "recovery_score"
TARGET_NAME = "boa_recuperacao"
TARGET_THRESHOLD = 66.0

FEATURES = [
    "day_strain",
    "sleep_hours",
    "sleep_efficiency",
    "hrv",
    "resting_heart_rate",
    "age",
    "weight_kg",
    "height_cm",
    "sleep_performance",
    "light_sleep_hours",
    "rem_sleep_hours",
    "deep_sleep_hours",
    "wake_ups",
    "time_to_fall_asleep_min",
    "hrv_baseline",
    "rhr_baseline",
    "respiratory_rate",
    "skin_temp_deviation",
]

FEATURE_LABELS = {
    "day_strain": "Esforço diário",
    "sleep_hours": "Horas de sono",
    "sleep_efficiency": "Eficiência do sono",
    "hrv": "Variabilidade cardíaca (HRV)",
    "resting_heart_rate": "Frequência cardíaca em repouso",
    "age": "Idade",
    "weight_kg": "Peso",
    "height_cm": "Altura",
    "sleep_performance": "Desempenho do sono",
    "light_sleep_hours": "Sono leve",
    "rem_sleep_hours": "Sono REM",
    "deep_sleep_hours": "Sono profundo",
    "wake_ups": "Despertares",
    "time_to_fall_asleep_min": "Tempo para adormecer",
    "hrv_baseline": "HRV de referência",
    "rhr_baseline": "FC de repouso de referência",
    "respiratory_rate": "Frequência respiratória",
    "skin_temp_deviation": "Desvio da temperatura da pele",
}

BASELINE_3VA_MODEL = "LightGBM"
BASELINE_3VA_ACCURACY = 0.658
BASELINE_3VA_CONTEXT = (
    "Resultado relatado na 3VA e usado apenas como referência descritiva, "
    "pois o protocolo final não é idêntico ao anterior."
)

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_PATH = DASHBOARD_DIR / "data" / "dataset.pkl"
DEFAULT_EXCEL_PATH = DASHBOARD_DIR / "whoop_fitness_dataset_100k.xlsx"
DEFAULT_OUTPUT_DIR = DASHBOARD_DIR / "resultados_carlos"
