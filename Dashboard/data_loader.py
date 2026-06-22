# ==================================================
# data_loader.py - VERSÃO COMPLETA COM FEATURES DERIVADAS
# ==================================================

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

class DataManager:
    """Gerencia TODO o carregamento e tratamento de dados"""
    
    # Mapeamento para tradução de colunas
    COLUMN_TRANSLATIONS = {
        # Identificação
        'user_id': 'ID Usuário',
        'date': 'Data',
        
        # Fitness & Atividade
        'fitness_level': 'Nível de Condicionamento',
        'gender': 'Gênero',
        'primary_sport': 'Esporte Principal',
        'activity_type': 'Tipo de Atividade',
        'workout_time_of_day': 'Horário do Treino',
        'activity_duration_min': 'Duração da Atividade (min)',
        'activity_duration_hours': 'Duração da Atividade (horas)',
        'calories_burned': 'Calorias Queimadas',
        'steps': 'Passos',
        
        # Recuperação & Sono
        'recovery_score': 'Pontuação de Recuperação',
        'sleep_hours': 'Horas de Sono',
        'sleep_efficiency': 'Eficiência do Sono',
        'sleep_performance': 'Performance do Sono',
        'light_sleep_hours': 'Sono Leve',
        'deep_sleep_hours': 'Sono Profundo',
        'rem_sleep_hours': 'Sono REM',
        'wake_ups': 'Despertares',
        'sleep_quality': 'Qualidade do Sono',
        'strain_per_sleep': 'Strain por Hora de Sono',
        'hrv_ratio': 'Variação de Frequência Cardíaca',
        'hrv_rhr_ratio': 'Variabilidade da Frequência Cardíaca/Frequência Cardíaca de Repouso',
        'rhr_ratio': 'Razão FCR',
        
        # Carga
        'day_strain': 'Tensão do Dia',
        'activity_strain': 'Carga da Atividade',
        
        # Cardíaco
        'hrv': 'VFC',
        'hrv_baseline': 'VFC Base',
        'resting_heart_rate': 'Frequência Cardíaca em Repouso',
        'rhr_baseline': 'Linha de base da FC em repouso',
        'avg_heart_rate': 'Frequência Cardíaca Média',
        'max_heart_rate': 'Frequência Cardíaca Máxima',
        
        # Corpo
        'respiratory_rate': 'Frequência Respiratória',
        'skin_temp_deviation': 'Desvio de Temperatura',
        
        # Performance
        'training_load': 'Carga de Treino',
        'fatigue_score': 'Score de Fadiga',
        'readiness_score': 'Score de Prontidão',
        
        # Demográfico
        'age': 'Idade',
        'weight_kg': 'Peso (kg)',
        'height_cm': 'Altura (cm)',
        
        # Treino
        'workout_completed': 'Treino concluído',
        'activity_calories': 'Calorias da atividade',
        'time_to_fall_asleep_min': 'Tempo para dormir (min)',
        
        # Zonas de FC
        'hr_zone_1_min': 'Minutos na Zona 1 de FC',
        'hr_zone_2_min': 'Minutos na Zona 2 de FC',
        'hr_zone_3_min': 'Minutos na Zona 3 de FC',
        'hr_zone_4_min': 'Minutos na Zona 4 de FC',
        'hr_zone_5_min': 'Minutos na Zona 5 de FC',
    }
    
    # Traduções de valores categóricos
    VALUE_TRANSLATIONS = {
        # Gênero
        'male': 'Masculino',
        'female': 'Feminino',
        
        # Nível
        'beginner': 'Iniciante',
        'intermediate': 'Intermediário',
        'advanced': 'Avançado',
        'elite': 'Elite',
        
        # Esportes
        'running': 'Corrida',
        'cycling': 'Ciclismo',
        'swimming': 'Natação',
        'weightlifting': 'Musculação',
        'weight training': 'Musculação',
        'yoga': 'Yoga',
        'hiit': 'HIIT',
        'crossfit': 'CrossFit',
        'walking': 'Caminhada',
        'rest day': 'Descanso',
        
        # Horários
        'morning': 'Manhã',
        'afternoon': 'Tarde',
        'evening': 'Noite',
        
        # Dias
        'monday': 'Segunda',
        'tuesday': 'Terça',
        'wednesday': 'Quarta',
        'thursday': 'Quinta',
        'friday': 'Sexta',
        'saturday': 'Sábado',
        'sunday': 'Domingo',
    }

    # Paleta de cores padrão do dashboard
    CORES = {
        # Fundos
        'background': '#0D0D0D',
        'card_bg': '#1A1A1A',
        'border': '#2A2A2A',
        
        # Textos
        'text': '#FFFFFF',
        'text_secondary': '#888888',
        
        # Destaques
        'accent': '#3B82F6',
        'success': '#10B981',
        'warning': '#F59E0B',
        'danger': '#EF4444',
        
        # Métricas principais
        'recovery': '#10B981',
        'sleep': '#A8D8EA',
        'hrv': '#D5B8E8',
        'strain': '#FAD7A0',
        'age': '#BDBDBD',
        
        # Gênero
        'masculino': '#A8D8EA',
        'feminino': '#F5B7B1',
        
        # Nível de Condicionamento
        'iniciante': '#A8D8EA',
        'intermediario': '#B5E3B5',
        'avancado': '#FAD7A0',
        'elite': '#D5B8E8',
        
        # Horários (escala de cinza)
        'manha': '#BDBDBD',
        'tarde': '#9E9E9E',
        'noite': '#757575',
        
        # Lista genérica para gráficos
        'chart_colors': ['#A8D8EA', '#B5E3B5', '#F5B7B1', '#FAD7A0', '#D5B8E8', '#A8E6CF']
    }

    def __init__(self, local_path="data/dataset.pkl"):
        self.local_path = Path(local_path)
        self.df_raw = None
        self.df_clean = None
        self.df_translated = None
        
        self.outliers = {}
        self.stats = {}
        self.cleaning_report = {}
        
    # ==========================================
    # 📊 FUNÇÕES DE LIMPEZA
    # ==========================================
    
    def _clean_respiratory_rate(self, series):
        """Limpa frequência respiratória (5-50 respirações/min)"""
        serie = pd.to_numeric(series, errors='coerce')
        mask_invalid = (serie < 5) | (serie > 50)
        if mask_invalid.any():
            self._add_outlier('respiratory', 'respiratory_rate', mask_invalid.sum(), serie[mask_invalid])
            serie.loc[mask_invalid] = pd.NA
        return serie
    
    def _clean_skin_temp(self, series):
        """Limpa desvio de temperatura (máximo 5°C)"""
        serie = pd.to_numeric(series, errors='coerce')
        # Primeiro, lidar com outliers extremos (valores > 1000)
        mask_extreme = serie > 1000
        if mask_extreme.any():
            self._add_outlier('skin_temp', 'skin_temp_extreme', mask_extreme.sum(), serie[mask_extreme])
            serie.loc[mask_extreme] = pd.NA
        
        # Depois, verificar desvio > 5°C
        mask_invalid = abs(serie) > 5
        if mask_invalid.any():
            self._add_outlier('skin_temp', 'skin_temp_deviation', mask_invalid.sum(), serie[mask_invalid])
            serie.loc[mask_invalid] = pd.NA
        return serie
    
    def _clean_activity_strain(self, series):
        """Limpa strain da atividade (0-21)"""
        serie = pd.to_numeric(series, errors='coerce')
        mask_invalid = (serie < 0) | (serie > 21)
        if mask_invalid.any():
            self._add_outlier('strain', 'activity_strain', mask_invalid.sum(), serie[mask_invalid])
            serie.loc[mask_invalid] = pd.NA
        return serie
    
    def _clean_time_to_fall_asleep(self, series):
        """Limpa tempo para dormir (5-60 min)"""
        serie = pd.to_numeric(series, errors='coerce')
        mask_invalid = (serie < 5) | (serie > 60)
        if mask_invalid.any():
            self._add_outlier('sleep', 'time_to_fall_asleep_min', mask_invalid.sum(), serie[mask_invalid])
            serie.loc[mask_invalid] = pd.NA
        return serie
    
    def _clean_wake_ups(self, series):
        """Limpa despertares (0-10)"""
        serie = pd.to_numeric(series, errors='coerce')
        mask_invalid = (serie < 0) | (serie > 10)
        if mask_invalid.any():
            self._add_outlier('sleep', 'wake_ups', mask_invalid.sum(), serie[mask_invalid])
            serie.loc[mask_invalid] = pd.NA
        return serie
    
    def _clean_activity_calories(self, series):
        """Limpa calorias da atividade (0-2000)"""
        serie = pd.to_numeric(series, errors='coerce')
        mask_invalid = (serie < 0) | (serie > 2000)
        if mask_invalid.any():
            self._add_outlier('calories', 'activity_calories', mask_invalid.sum(), serie[mask_invalid])
            serie.loc[mask_invalid] = pd.NA
        return serie
    
    def _clean_weight(self, series):
        """Limpa peso (20-300 kg)"""
        serie = pd.to_numeric(series, errors='coerce')
        mask_invalid = (serie < 20) | (serie > 300)
        if mask_invalid.any():
            self._add_outlier('body', 'weight_kg', mask_invalid.sum(), serie[mask_invalid])
            serie.loc[mask_invalid] = pd.NA
        return serie
    
    def _clean_height(self, series):
        """Limpa altura (100-250 cm)"""
        serie = pd.to_numeric(series, errors='coerce')
        mask_invalid = (serie < 100) | (serie > 250)
        if mask_invalid.any():
            self._add_outlier('body', 'height_cm', mask_invalid.sum(), serie[mask_invalid])
            serie.loc[mask_invalid] = pd.NA
        return serie
    
    def _clean_hr_zones(self, series):
        """Limpa zonas de FC (não podem ser negativas e nem > 24h)"""
        serie = pd.to_numeric(series, errors='coerce')
        mask_invalid = (serie < 0) | (serie > 1440)  # 24 horas em minutos
        if mask_invalid.any():
            self._add_outlier('hr_zones', series.name, mask_invalid.sum(), serie[mask_invalid])
            serie.loc[mask_invalid] = pd.NA
        return serie
    
    def _clean_workout_completed(self, series):
        """Limpa workout_completed (deve ser 0 ou 1)"""
        serie = pd.to_numeric(series, errors='coerce')
        mask_invalid = ~serie.isin([0, 1])
        if mask_invalid.any():
            self._add_outlier('workout', 'workout_completed', mask_invalid.sum(), serie[mask_invalid])
            serie.loc[mask_invalid] = pd.NA
        return serie
    
    def _clean_hrv(self, series):
        """Limpa HRV (10-500 ms)"""
        serie = pd.to_numeric(series, errors='coerce')
        mask_high = serie > 500
        if mask_high.any():
            self._add_outlier('hrv', 'high', mask_high.sum(), serie[mask_high])
            serie.loc[mask_high] = pd.NA
        mask_low = serie < 10
        if mask_low.any():
            self._add_outlier('hrv', 'low', mask_low.sum(), serie[mask_low])
            serie.loc[mask_low] = pd.NA
        return serie
    
    def _clean_sleep(self, series, col_name):
        """Limpa horas de sono (0-24)"""
        serie = pd.to_numeric(series, errors='coerce')
        mask_invalid = (serie < 0) | (serie > 24)
        if mask_invalid.any():
            self._add_outlier('sleep', col_name, mask_invalid.sum(), serie[mask_invalid])
            serie.loc[mask_invalid] = pd.NA
        return serie
    
    def _clean_recovery(self, series):
        """Limpa recovery score (0-100)"""
        serie = pd.to_numeric(series, errors='coerce')
        mask_invalid = (serie < 0) | (serie > 100)
        if mask_invalid.any():
            self._add_outlier('recovery', 'recovery_score', mask_invalid.sum(), serie[mask_invalid])
            serie.loc[mask_invalid] = pd.NA
        return serie
    
    def _clean_heart_rate(self, series, col_name):
        """Limpa frequência cardíaca (30-220 bpm)"""
        serie = pd.to_numeric(series, errors='coerce')
        mask_invalid = (serie < 30) | (serie > 220)
        if mask_invalid.any():
            self._add_outlier('heart_rate', col_name, mask_invalid.sum(), serie[mask_invalid])
            serie.loc[mask_invalid] = pd.NA
        return serie
    
    def _clean_age(self, series):
        """Limpa idade (10-100 anos)"""
        serie = pd.to_numeric(series, errors='coerce')
        mask_invalid = (serie < 10) | (serie > 100)
        if mask_invalid.any():
            self._add_outlier('demographic', 'age', mask_invalid.sum(), serie[mask_invalid])
            serie.loc[mask_invalid] = pd.NA
        return serie
    
    def _clean_calories(self, series):
        """Limpa calorias (0-5000)"""
        serie = pd.to_numeric(series, errors='coerce')
        mask_invalid = (serie < 0) | (serie > 5000)
        if mask_invalid.any():
            self._add_outlier('energy', 'calories_burned', mask_invalid.sum(), serie[mask_invalid])
            serie.loc[mask_invalid] = pd.NA
        return serie
    
    def _clean_steps(self, series):
        """Limpa passos (0-50000)"""
        serie = pd.to_numeric(series, errors='coerce')
        mask_invalid = (serie < 0) | (serie > 50000)
        if mask_invalid.any():
            self._add_outlier('activity', 'steps', mask_invalid.sum(), serie[mask_invalid])
            serie.loc[mask_invalid] = pd.NA
        return serie
    
    def _clean_activity_duration(self, series):
        """Limpa duração da atividade (0-720 min)"""
        serie = pd.to_numeric(series, errors='coerce')
        mask_invalid = (serie < 0) | (serie > 720)
        if mask_invalid.any():
            self._add_outlier('activity', 'activity_duration_min', mask_invalid.sum(), serie[mask_invalid])
            serie.loc[mask_invalid] = pd.NA
        return serie
    
    def _convert_to_hours(self, series):
        """Converte minutos para horas"""
        serie = pd.to_numeric(series, errors='coerce')
        return serie / 60
    
    def _add_outlier(self, category, field, count, values):
        if category not in self.outliers:
            self.outliers[category] = {}
        self.outliers[category][field] = {
            'count': count,
            'min': float(values.min()),
            'max': float(values.max()),
            'mean': float(values.mean()),
            'values': values.tolist()[:10]
        }
    
    # ==========================================
    # 📊 FEATURES DERIVADAS - NOVO!
    # ==========================================
    
    def _criar_features_derivadas(self, df):
        """
        Cria features derivadas que serão usadas em todo o dashboard
        Essas features são criadas a partir das colunas existentes
        """
        print("\n🔧 Criando features derivadas...")
        df_new = df.copy()
        features_criadas = []
        
        # Qualidade do Sono
        if 'sleep_hours' in df.columns and 'sleep_efficiency' in df.columns:
            df_new['sleep_quality'] = df['sleep_hours'] * (df['sleep_efficiency'] / 100)
            features_criadas.append('sleep_quality')
            print("   ✅ sleep_quality criada")
        
        # Strain por Hora de Sono
        if 'day_strain' in df.columns and 'sleep_hours' in df.columns:
            df_new['strain_per_sleep'] = df['day_strain'] / (df['sleep_hours'] + 0.1)
            features_criadas.append('strain_per_sleep')
            print("   ✅ strain_per_sleep criada")
        
        # HRV Ratio (HRV atual vs baseline)
        if 'hrv' in df.columns and 'hrv_baseline' in df.columns:
            df_new['hrv_ratio'] = df['hrv'] / (df['hrv_baseline'] + 1)
            features_criadas.append('hrv_ratio')
            print("   ✅ hrv_ratio criada")
        
        # RHR Ratio (FCR atual vs baseline)
        if 'resting_heart_rate' in df.columns and 'rhr_baseline' in df.columns:
            df_new['rhr_ratio'] = df['resting_heart_rate'] / (df['rhr_baseline'] + 1)
            features_criadas.append('rhr_ratio')
            print("   ✅ rhr_ratio criada")
        
        # HRV / RHR Ratio
        if 'hrv' in df.columns and 'resting_heart_rate' in df.columns:
            df_new['hrv_rhr_ratio'] = df['hrv'] / (df['resting_heart_rate'] + 1)
            features_criadas.append('hrv_rhr_ratio')
            print("   ✅ hrv_rhr_ratio criada")
        
        print(f"   ✅ Total de {len(features_criadas)} features derivadas criadas")
        return df_new, features_criadas
    
    # ==========================================
    # 📊 PRÉ-PROCESSAMENTO (ATUALIZADO)
    # ==========================================
    
    def _preprocess_dataframe(self, df):
        print("\n" + "="*50)
        print("🔧 INICIANDO PRÉ-PROCESSAMENTO")
        print("="*50)
        
        df_processed = df.copy()
        initial_rows = len(df_processed)
        
        # Datas
        if 'date' in df_processed.columns:
            df_processed['date'] = pd.to_datetime(df_processed['date'], errors='coerce')
            print("✅ Datas convertidas")
        
        # ==========================================
        # LIMPEZA DE COLUNAS
        # ==========================================
        
        if 'hrv' in df_processed.columns:
            print("\n📊 Limpando HRV...")
            df_processed['hrv'] = self._clean_hrv(df_processed['hrv'])
        
        if 'hrv_baseline' in df_processed.columns:
            print("\n📊 Limpando HRV Baseline...")
            df_processed['hrv_baseline'] = self._clean_hrv(df_processed['hrv_baseline'])
        
        sleep_cols = ['sleep_hours', 'light_sleep_hours', 'deep_sleep_hours', 'rem_sleep_hours']
        for col in sleep_cols:
            if col in df_processed.columns:
                print(f"\n📊 Limpando {col}...")
                df_processed[col] = self._clean_sleep(df_processed[col], col)
        
        if 'recovery_score' in df_processed.columns:
            print("\n📊 Limpando Recovery Score...")
            df_processed['recovery_score'] = self._clean_recovery(df_processed['recovery_score'])
        
        hr_cols = ['resting_heart_rate', 'avg_heart_rate', 'max_heart_rate']
        for col in hr_cols:
            if col in df_processed.columns:
                print(f"\n📊 Limpando {col}...")
                df_processed[col] = self._clean_heart_rate(df_processed[col], col)
        
        if 'age' in df_processed.columns:
            print("\n📊 Limpando Idade...")
            df_processed['age'] = self._clean_age(df_processed['age'])
        
        if 'calories_burned' in df_processed.columns:
            print("\n📊 Limpando Calorias...")
            df_processed['calories_burned'] = self._clean_calories(df_processed['calories_burned'])
        
        if 'steps' in df_processed.columns:
            print("\n📊 Limpando Passos...")
            df_processed['steps'] = self._clean_steps(df_processed['steps'])
        
        if 'activity_duration_min' in df_processed.columns:
            print("\n📊 Limpando Duração da Atividade...")
            df_processed['activity_duration_min'] = self._clean_activity_duration(df_processed['activity_duration_min'])
            df_processed['activity_duration_hours'] = self._convert_to_hours(df_processed['activity_duration_min'])
            print("   ✅ Criada coluna 'activity_duration_hours'")
        
        if 'respiratory_rate' in df_processed.columns:
            print("\n📊 Limpando Frequência Respiratória...")
            df_processed['respiratory_rate'] = self._clean_respiratory_rate(df_processed['respiratory_rate'])
        
        if 'skin_temp_deviation' in df_processed.columns:
            print("\n📊 Limpando Desvio de Temperatura...")
            df_processed['skin_temp_deviation'] = self._clean_skin_temp(df_processed['skin_temp_deviation'])
        
        if 'activity_strain' in df_processed.columns:
            print("\n📊 Limpando Strain da Atividade...")
            df_processed['activity_strain'] = self._clean_activity_strain(df_processed['activity_strain'])
        
        if 'time_to_fall_asleep_min' in df_processed.columns:
            print("\n📊 Limpando Tempo para Dormir...")
            df_processed['time_to_fall_asleep_min'] = self._clean_time_to_fall_asleep(df_processed['time_to_fall_asleep_min'])
        
        if 'wake_ups' in df_processed.columns:
            print("\n📊 Limpando Despertares...")
            df_processed['wake_ups'] = self._clean_wake_ups(df_processed['wake_ups'])
        
        if 'activity_calories' in df_processed.columns:
            print("\n📊 Limpando Calorias da Atividade...")
            df_processed['activity_calories'] = self._clean_activity_calories(df_processed['activity_calories'])
        
        if 'weight_kg' in df_processed.columns:
            print("\n📊 Limpando Peso...")
            df_processed['weight_kg'] = self._clean_weight(df_processed['weight_kg'])
        
        if 'height_cm' in df_processed.columns:
            print("\n📊 Limpando Altura...")
            df_processed['height_cm'] = self._clean_height(df_processed['height_cm'])
        
        hr_zone_cols = ['hr_zone_1_min', 'hr_zone_2_min', 'hr_zone_3_min', 'hr_zone_4_min', 'hr_zone_5_min']
        for col in hr_zone_cols:
            if col in df_processed.columns:
                print(f"\n📊 Limpando {col}...")
                df_processed[col] = self._clean_hr_zones(df_processed[col])
        
        if 'workout_completed' in df_processed.columns:
            print("\n📊 Limpando Treino Concluído...")
            df_processed['workout_completed'] = self._clean_workout_completed(df_processed['workout_completed'])
        
        # ==========================================
        # CRIAR FEATURES DERIVADAS - NOVO!
        # ==========================================
        df_processed, features_criadas = self._criar_features_derivadas(df_processed)
        
        # Remover linhas completamente vazias
        df_processed = df_processed.dropna(how='all')
        
        # Remover duplicatas
        duplicates = df_processed.duplicated().sum()
        if duplicates > 0:
            df_processed = df_processed.drop_duplicates()
            print(f"\n🗑️ Removidas {duplicates} linhas duplicadas")
        
        final_rows = len(df_processed)
        print(f"\n📊 Resumo do pré-processamento:")
        print(f"   Registros iniciais: {initial_rows:,}")
        print(f"   Registros finais: {final_rows:,}")
        print(f"   Removidos: {initial_rows - final_rows:,} ({((initial_rows - final_rows)/initial_rows*100):.1f}%)")
        print(f"   Features derivadas criadas: {len(features_criadas)}")
        
        return df_processed
    
    # ==========================================
    # 📊 TRADUÇÕES
    # ==========================================
    
    def _translate_dataframe(self, df):
        df_translated = df.copy()
        categorical_cols = ['gender', 'fitness_level', 'primary_sport', 
                           'activity_type', 'workout_time_of_day']
        for col in categorical_cols:
            if col in df_translated.columns:
                df_translated[col] = df_translated[col].apply(
                    lambda x: self.VALUE_TRANSLATIONS.get(str(x).lower(), x) if pd.notna(x) else x
                )
        return df_translated
    
    def get_translated_df(self):
        if self.df_translated is None and self.df_clean is not None:
            self.df_translated = self._translate_dataframe(self.df_clean)
        return self.df_translated
    
    def traduzir_coluna(self, nome_coluna):
        return self.COLUMN_TRANSLATIONS.get(nome_coluna, nome_coluna)
    
    def get_cores(self):
        return self.CORES
    
    # ==========================================
    # 📊 MÉTRICAS E ESTATÍSTICAS
    # ==========================================
    
    def _calculate_stats(self, df):
        self.stats = {
            'total_records': len(df),
            'total_users': df['user_id'].nunique() if 'user_id' in df.columns else 0,
            'date_range': {
                'start': df['date'].min() if 'date' in df.columns else None,
                'end': df['date'].max() if 'date' in df.columns else None
            },
            'numeric_columns': {}
        }
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.stats['numeric_columns'][col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'null_count': int(df[col].isnull().sum()),
                'null_percentage': float(df[col].isnull().sum() / len(df) * 100)
            }
    
    # ==========================================
    # 📥 CARREGAMENTO PRINCIPAL
    # ==========================================
    
    def load_data(self, force_reload=True):
        if not force_reload and self.local_path.exists():
            print(f"📦 Carregando dados do cache: {self.local_path}")
            try:
                with open(self.local_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.df_clean = cache_data['df_clean']
                    self.outliers = cache_data.get('outliers', {})
                    self.stats = cache_data.get('stats', {})
                    self.df_translated = None
                print("✅ Cache carregado com sucesso")
                return self.df_clean
            except Exception as e:
                print(f"⚠️ Erro ao carregar cache: {e}")
                print("🔄 Recarregando dados originais...")
        
        print("\n📂 Carregando dados do Excel...")
        excel_files = list(Path(".").rglob("*.xlsx"))
        if not excel_files:
            print("❌ Nenhum arquivo XLSX encontrado!")
            return None
        
        excel_file = excel_files[0]
        print(f"📄 Arquivo encontrado: {excel_file.name}")
        
        try:
            self.df_raw = pd.read_excel(excel_file, engine='openpyxl')
            print(f"✅ {len(self.df_raw):,} registros carregados")
            self.df_clean = self._preprocess_dataframe(self.df_raw)
            self._calculate_stats(self.df_clean)
            self.local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.local_path, 'wb') as f:
                pickle.dump({
                    'df_clean': self.df_clean,
                    'outliers': self.outliers,
                    'stats': self.stats
                }, f)
            print(f"\n💾 Cache salvo em: {self.local_path}")
            self._print_report()
            return self.df_clean
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _print_report(self):
        print("\n" + "="*50)
        print("📊 RELATÓRIO COMPLETO")
        print("="*50)
        print(f"\n✅ DADOS PRONTOS:")
        print(f"   Registros: {self.stats['total_records']:,}")
        print(f"   Usuários: {self.stats['total_users']:,}")
        if self.stats['date_range']['start']:
            print(f"   Período: {self.stats['date_range']['start'].date()} a {self.stats['date_range']['end'].date()}")
        
        # Mostrar features derivadas criadas
        print("\n🔧 FEATURES DERIVADAS CRIADAS:")
        features_derivadas = ['sleep_quality', 'strain_per_sleep', 'hrv_ratio', 'rhr_ratio', 'hrv_rhr_ratio']
        derivadas_existentes = [f for f in features_derivadas if f in self.df_clean.columns]
        if derivadas_existentes:
            for f in derivadas_existentes:
                print(f"   ✅ {f} ({self.traduzir_coluna(f)})")
        else:
            print("   ⚠️ Nenhuma feature derivada criada")
        
        if self.outliers:
            print("\n⚠️ OUTLIERS REMOVIDOS:")
            total_outliers = 0
            for category, fields in self.outliers.items():
                print(f"\n   {category.upper()}:")
                for field, info in fields.items():
                    print(f"      - {field}: {info['count']} valores")
                    print(f"        Range: {info['min']:.1f} - {info['max']:.1f}")
                    total_outliers += info['count']
            print(f"\n   TOTAL: {total_outliers} outliers removidos")
        
        print("\n📈 QUALIDADE DOS DADOS:")
        for col, stats in self.stats['numeric_columns'].items():
            if stats['null_percentage'] > 0:
                print(f"   {col}: {stats['null_percentage']:.1f}% valores nulos")
    
    # ==========================================
    # 📤 MÉTODOS PÚBLICOS
    # ==========================================
    
    def get_clean_df(self):
        return self.df_clean
    
    def get_raw_df(self):
        return self.df_raw
    
    def get_stats(self):
        return self.stats
    
    def get_outliers(self):
        return self.outliers
    
    def get_column_names(self, translated=False):
        if self.df_clean is None:
            return []
        columns = self.df_clean.columns.tolist()
        if translated:
            return [self.traduzir_coluna(col) for col in columns]
        return columns
    
    def get_column_info(self, column):
        if self.df_clean is None or column not in self.df_clean.columns:
            return None
        col_data = self.df_clean[column].dropna()
        info = {
            'name': column,
            'name_translated': self.traduzir_coluna(column),
            'dtype': str(self.df_clean[column].dtype),
            'count': len(col_data),
            'null_count': self.df_clean[column].isnull().sum(),
            'unique_count': col_data.nunique()
        }
        if pd.api.types.is_numeric_dtype(col_data):
            info.update({
                'mean': float(col_data.mean()),
                'median': float(col_data.median()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'max': float(col_data.max())
            })
        else:
            info['top_values'] = col_data.value_counts().head(5).to_dict()
        return info
    
    def get_features_derivadas(self):
        """Retorna a lista de features derivadas criadas"""
        features_derivadas = ['sleep_quality', 'strain_per_sleep', 'hrv_ratio', 'rhr_ratio', 'hrv_rhr_ratio']
        if self.df_clean is None:
            return []
        return [f for f in features_derivadas if f in self.df_clean.columns]


# Instância global única
data_manager = DataManager()