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
        
        'workout_completed': 'Treino concluído',
        'activity_calories': 'Calorias da atividade',
        'hr_zone_2_min': 'Minutos na Zona 2 de FC',
        'hr_zone_3_min': 'Minutos na Zona 3 de FC',
        'hr_zone_1_min': 'Minutos na Zona 1 de FC',
        'hr_zone_4_min': 'Minutos na Zona 4 de FC',
        'hr_zone_5_min': 'Minutos na Zona 5 de FC',
        'weight_kg': 'Peso (kg)',
        'height_cm': 'Altura (cm)',
        'time_to_fall_asleep_min': 'Tempo para dormir (min)',
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
        self.df_raw = None  # Dados brutos originais
        self.df_clean = None  # Dados tratados
        self.df_translated = None  # Dados com tradução (apenas para visualização)
        
        # Estatísticas e outliers
        self.outliers = {}
        self.stats = {}
        self.cleaning_report = {}
        
    # ==========================================
    # 📊 LIMPEZA DE DADOS (Métodos Privados)
    # ==========================================
    
    def _clean_hrv(self, series):
        """Limpa valores de HRV"""
        serie = pd.to_numeric(series, errors='coerce')
        
        # Remover valores absurdos (>500 ms)
        mask_high = serie > 500
        if mask_high.any():
            self._add_outlier('hrv', 'high', mask_high.sum(), serie[mask_high])
            serie.loc[mask_high] = pd.NA
        
        # Remover valores muito baixos (<10 ms)
        mask_low = serie < 10
        if mask_low.any():
            self._add_outlier('hrv', 'low', mask_low.sum(), serie[mask_low])
            serie.loc[mask_low] = pd.NA
        
        return serie
    
    def _clean_sleep(self, series, col_name):
        """Limpa valores de horas de sono"""
        serie = pd.to_numeric(series, errors='coerce')
        
        # Limites realistas: 0-24 horas
        mask_invalid = (serie < 0) | (serie > 24)
        if mask_invalid.any():
            self._add_outlier('sleep', col_name, mask_invalid.sum(), serie[mask_invalid])
            serie.loc[mask_invalid] = pd.NA
        
        return serie
    
    def _clean_recovery(self, series):
        """Limpa valores de recovery score"""
        serie = pd.to_numeric(series, errors='coerce')
        
        # Recovery score deve estar entre 0 e 100
        mask_invalid = (serie < 0) | (serie > 100)
        if mask_invalid.any():
            self._add_outlier('recovery', 'recovery_score', mask_invalid.sum(), serie[mask_invalid])
            serie.loc[mask_invalid] = pd.NA
        
        return serie
    
    def _clean_heart_rate(self, series, col_name):
        """Limpa valores de frequência cardíaca"""
        serie = pd.to_numeric(series, errors='coerce')
        
        # Limites fisiológicos: 30-220 bpm
        mask_invalid = (serie < 30) | (serie > 220)
        if mask_invalid.any():
            self._add_outlier('heart_rate', col_name, mask_invalid.sum(), serie[mask_invalid])
            serie.loc[mask_invalid] = pd.NA
        
        return serie
    
    def _clean_age(self, series):
        """Limpa valores de idade"""
        serie = pd.to_numeric(series, errors='coerce')
        
        # Limites: 10-100 anos
        mask_invalid = (serie < 10) | (serie > 100)
        if mask_invalid.any():
            self._add_outlier('demographic', 'age', mask_invalid.sum(), serie[mask_invalid])
            serie.loc[mask_invalid] = pd.NA
        
        return serie
    
    def _clean_calories(self, series):
        """Limpa valores de calorias"""
        serie = pd.to_numeric(series, errors='coerce')
        
        # Limites: 0-5000 calorias por treino
        mask_invalid = (serie < 0) | (serie > 5000)
        if mask_invalid.any():
            self._add_outlier('energy', 'calories_burned', mask_invalid.sum(), serie[mask_invalid])
            serie.loc[mask_invalid] = pd.NA
        
        return serie
    
    def _clean_steps(self, series):
        """Limpa valores de passos"""
        serie = pd.to_numeric(series, errors='coerce')
        
        # Limites: 0-50000 passos por dia
        mask_invalid = (serie < 0) | (serie > 50000)
        if mask_invalid.any():
            self._add_outlier('activity', 'steps', mask_invalid.sum(), serie[mask_invalid])
            serie.loc[mask_invalid] = pd.NA
        
        return serie
    
    def _clean_activity_duration(self, series):
        """Limpa e converte duração da atividade"""
        serie = pd.to_numeric(series, errors='coerce')
        
        # Limites: 0-720 minutos (12 horas)
        mask_invalid = (serie < 0) | (serie > 720)
        if mask_invalid.any():
            self._add_outlier('activity', 'activity_duration_min', mask_invalid.sum(), serie[mask_invalid])
            serie.loc[mask_invalid] = pd.NA
        
        return serie
    
    def _convert_to_hours(self, series):
        """Converte minutos para horas se necessário"""
        serie = pd.to_numeric(series, errors='coerce')
        
        # Se o valor máximo for > 24, provavelmente está em minutos
        if not serie.empty and serie.max() > 24:
            horas = serie / 60
            return horas
        return serie
    
    def _add_outlier(self, category, field, count, values):
        """Adiciona outlier ao relatório"""
        if category not in self.outliers:
            self.outliers[category] = {}
        
        self.outliers[category][field] = {
            'count': count,
            'min': float(values.min()),
            'max': float(values.max()),
            'mean': float(values.mean()),
            'values': values.tolist()[:10]  # Guarda só os 10 primeiros
        }
    
    # ==========================================
    # 📊 PRÉ-PROCESSAMENTO
    # ==========================================
    
    def _preprocess_dataframe(self, df):
        """Aplica todo o pré-processamento"""
        print("\n" + "="*50)
        print("🔧 INICIANDO PRÉ-PROCESSAMENTO")
        print("="*50)
        
        df_processed = df.copy()
        initial_rows = len(df_processed)
        
        # 1. Converter datas
        if 'date' in df_processed.columns:
            df_processed['date'] = pd.to_datetime(df_processed['date'], errors='coerce')
            print("✅ Datas convertidas")
        
        # 2. Limpar HRV
        if 'hrv' in df_processed.columns:
            print("\n📊 Limpando HRV...")
            df_processed['hrv'] = self._clean_hrv(df_processed['hrv'])
        
        if 'hrv_baseline' in df_processed.columns:
            df_processed['hrv_baseline'] = self._clean_hrv(df_processed['hrv_baseline'])
        
        # 3. Limpar horas de sono
        sleep_cols = ['sleep_hours', 'light_sleep_hours', 'deep_sleep_hours', 'rem_sleep_hours']
        for col in sleep_cols:
            if col in df_processed.columns:
                print(f"\n📊 Limpando {col}...")
                df_processed[col] = self._clean_sleep(df_processed[col], col)
        
        # 4. Limpar recovery score
        if 'recovery_score' in df_processed.columns:
            print("\n📊 Limpando Recovery Score...")
            df_processed['recovery_score'] = self._clean_recovery(df_processed['recovery_score'])
        
        # 5. Limpar frequência cardíaca
        hr_cols = ['resting_heart_rate', 'avg_heart_rate', 'max_heart_rate']
        for col in hr_cols:
            if col in df_processed.columns:
                print(f"\n📊 Limpando {col}...")
                df_processed[col] = self._clean_heart_rate(df_processed[col], col)
        
        # 6. Limpar idade
        if 'age' in df_processed.columns:
            print("\n📊 Limpando Idade...")
            df_processed['age'] = self._clean_age(df_processed['age'])
        
        # 7. Limpar calorias
        if 'calories_burned' in df_processed.columns:
            print("\n📊 Limpando Calorias...")
            df_processed['calories_burned'] = self._clean_calories(df_processed['calories_burned'])
        
        # 8. Limpar passos
        if 'steps' in df_processed.columns:
            print("\n📊 Limpando Passos...")
            df_processed['steps'] = self._clean_steps(df_processed['steps'])
        
        # 9. Limpar duração da atividade e criar coluna em horas
        if 'activity_duration_min' in df_processed.columns:
            print("\n📊 Limpando Duração da Atividade...")
            df_processed['activity_duration_min'] = self._clean_activity_duration(df_processed['activity_duration_min'])
            df_processed['activity_duration_hours'] = self._convert_to_hours(df_processed['activity_duration_min'])
            print("   ✅ Criada coluna 'activity_duration_hours'")
        
        # 10. Remover linhas completamente vazias
        df_processed = df_processed.dropna(how='all')
        
        # 11. Remover duplicatas
        duplicates = df_processed.duplicated().sum()
        if duplicates > 0:
            df_processed = df_processed.drop_duplicates()
            print(f"\n🗑️ Removidas {duplicates} linhas duplicadas")
        
        # Estatísticas finais
        final_rows = len(df_processed)
        print(f"\n📊 Resumo do pré-processamento:")
        print(f"   Registros iniciais: {initial_rows:,}")
        print(f"   Registros finais: {final_rows:,}")
        print(f"   Removidos: {initial_rows - final_rows:,} ({((initial_rows - final_rows)/initial_rows*100):.1f}%)")
        
        return df_processed
    
    # ==========================================
    # 📊 TRADUÇÕES
    # ==========================================
    
    def _translate_dataframe(self, df):
        """Cria versão traduzida do DataFrame (apenas para visualização)"""
        df_translated = df.copy()
        
        # Traduzir colunas categóricas
        categorical_cols = ['gender', 'fitness_level', 'primary_sport', 
                           'activity_type', 'workout_time_of_day']
        
        for col in categorical_cols:
            if col in df_translated.columns:
                df_translated[col] = df_translated[col].apply(
                    lambda x: self.VALUE_TRANSLATIONS.get(str(x).lower(), x) if pd.notna(x) else x
                )
        
        return df_translated
    
    def get_translated_df(self):
        """Retorna DataFrame com valores traduzidos (para visualização)"""
        if self.df_translated is None and self.df_clean is not None:
            self.df_translated = self._translate_dataframe(self.df_clean)
        return self.df_translated
    
    def traduzir_coluna(self, nome_coluna):
        """Traduz nome de coluna para português"""
        return self.COLUMN_TRANSLATIONS.get(nome_coluna, nome_coluna)
    
    def get_cores(self):
        """Retorna a paleta de cores padrão do dashboard"""
        return self.CORES
    
    # ==========================================
    # 📊 MÉTRICAS E ESTATÍSTICAS
    # ==========================================
    
    def _calculate_stats(self, df):
        """Calcula estatísticas descritivas"""
        self.stats = {
            'total_records': len(df),
            'total_users': df['user_id'].nunique() if 'user_id' in df.columns else 0,
            'date_range': {
                'start': df['date'].min() if 'date' in df.columns else None,
                'end': df['date'].max() if 'date' in df.columns else None
            },
            'numeric_columns': {}
        }
        
        # Estatísticas para colunas numéricas
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
    
    def load_data(self, force_reload=False):
        """
        Carrega e processa todos os dados
        
        Args:
            force_reload: Se True, ignora cache e recarrega do Excel
            
        Returns:
            DataFrame tratado e pronto para uso
        """
        # Tentar carregar do cache
        if not force_reload and self.local_path.exists():
            print(f"📦 Carregando dados do cache: {self.local_path}")
            try:
                with open(self.local_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.df_clean = cache_data['df_clean']
                    self.outliers = cache_data.get('outliers', {})
                    self.stats = cache_data.get('stats', {})
                    self.df_translated = None  # Recriar na demanda
                print("✅ Cache carregado com sucesso")
                return self.df_clean
            except Exception as e:
                print(f"⚠️ Erro ao carregar cache: {e}")
                print("🔄 Recarregando dados originais...")
        
        # Carregar do Excel
        print("\n📂 Carregando dados do Excel...")
        
        # Procurar arquivo Excel
        excel_files = list(Path(".").rglob("*.xlsx"))
        if not excel_files:
            print("❌ Nenhum arquivo XLSX encontrado!")
            return None
        
        excel_file = excel_files[0]
        print(f"📄 Arquivo encontrado: {excel_file.name}")
        
        try:
            # Carregar dados brutos
            self.df_raw = pd.read_excel(excel_file, engine='openpyxl')
            print(f"✅ {len(self.df_raw):,} registros carregados")
            
            # Aplicar pré-processamento completo
            self.df_clean = self._preprocess_dataframe(self.df_raw)
            
            # Calcular estatísticas
            self._calculate_stats(self.df_clean)
            
            # Salvar cache
            self.local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.local_path, 'wb') as f:
                pickle.dump({
                    'df_clean': self.df_clean,
                    'outliers': self.outliers,
                    'stats': self.stats
                }, f)
            print(f"\n💾 Cache salvo em: {self.local_path}")
            
            # Gerar relatório
            self._print_report()
            
            return self.df_clean
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _print_report(self):
        """Imprime relatório completo do processamento"""
        print("\n" + "="*50)
        print("📊 RELATÓRIO COMPLETO")
        print("="*50)
        
        print(f"\n✅ DADOS PRONTOS:")
        print(f"   Registros: {self.stats['total_records']:,}")
        print(f"   Usuários: {self.stats['total_users']:,}")
        
        if self.stats['date_range']['start']:
            print(f"   Período: {self.stats['date_range']['start'].date()} a {self.stats['date_range']['end'].date()}")
        
        # Relatório de outliers
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
        
        # Qualidade dos dados
        print("\n📈 QUALIDADE DOS DADOS:")
        for col, stats in self.stats['numeric_columns'].items():
            if stats['null_percentage'] > 0:
                print(f"   {col}: {stats['null_percentage']:.1f}% valores nulos")
    
    # ==========================================
    # 📤 MÉTODOS PÚBLICOS PARA ACESSO
    # ==========================================
    
    def get_clean_df(self):
        """Retorna DataFrame completamente tratado (em inglês)"""
        return self.df_clean
    
    def get_raw_df(self):
        """Retorna DataFrame bruto original"""
        return self.df_raw
    
    def get_stats(self):
        """Retorna estatísticas calculadas"""
        return self.stats
    
    def get_outliers(self):
        """Retorna relatório de outliers"""
        return self.outliers
    
    def get_column_names(self, translated=False):
        """Retorna lista de colunas disponíveis"""
        if self.df_clean is None:
            return []
        
        columns = self.df_clean.columns.tolist()
        if translated:
            return [self.traduzir_coluna(col) for col in columns]
        return columns
    
    def get_column_info(self, column):
        """Retorna informações detalhadas de uma coluna"""
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
        
        # Adicionar estatísticas se for numérico
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


# Instância global única
data_manager = DataManager()