import pandas as pd
from pathlib import Path
import time


class DataManager:
    """Gerencia carregamento portátil e otimizado dos dados"""

    def __init__(self, excel_path=None):

        self.df = None
        self.filtered_df = None  # 🔥 Adicionado para suportar filtros
        self.filters_applied = False  # 🔥 Flag para saber se filtros foram aplicados

        # 🔥 se não passar path, procura automaticamente
        self.excel_path = excel_path or self._find_excel()

        self.parquet_path = Path("data/dataset.parquet")

    # =====================================================
    # CARREGAMENTO PRINCIPAL
    # =====================================================
    def load_data(self, force_reload=False):
        """
        Carrega os dados (Excel ou Parquet)
        Retorna o DataFrame carregado
        """

        inicio = time.time()

        excel_file = self._find_excel()

        if not excel_file:
            raise FileNotFoundError(
                "Nenhum arquivo .xlsx encontrado no projeto."
            )

        # =========================================
        # USAR PARQUET SE EXISTIR
        # =========================================
        if (
            self.parquet_path.exists()
            and not force_reload
            and self._cache_valido(excel_file)
        ):
            print("📁 Carregando Parquet (cache)...")

            self.df = pd.read_parquet(self.parquet_path)
            self.filtered_df = self.df.copy()  # 🔥 Inicializa filtered_df

            self._log(inicio, "Parquet")

            return self.df

        # =========================================
        # LER EXCEL
        # =========================================
        print(f"📄 Lendo Excel: {excel_file}")

        self.df = pd.read_excel(
            excel_file,
            engine="openpyxl"
        )

        self._preprocess()

        self._save_cache()
        
        # 🔥 Inicializa filtered_df com os dados carregados
        self.filtered_df = self.df.copy()
        self.filters_applied = False

        self._log(inicio, "Excel → Parquet")

        return self.df

    # =====================================================
    # 🔥 NOVO MÉTODO: GET CURRENT DF
    # =====================================================
    def get_current_df(self):
        """
        Retorna o DataFrame atual (com filtros aplicados ou original)
        Útil para callbacks que precisam dos dados mais recentes
        """
        if self.filtered_df is not None and self.filters_applied:
            return self.filtered_df
        elif self.df is not None:
            return self.df
        else:
            # Se ainda não carregou, carrega agora
            return self.load_data()
    
    # =====================================================
    # 🔥 NOVO MÉTODO: APPLY FILTERS
    # =====================================================
    def apply_filters(self, filter_dict=None, query_string=None):
        """
        Aplica filtros aos dados
        
        Parâmetros:
        - filter_dict: dicionário com coluna: valor para filtrar
        - query_string: string de query estilo pandas
        
        Exemplo:
        filter_dict = {'gender': 'Male', 'fitness_level': 'High'}
        ou
        query_string = "recovery_score > 70 and day_strain < 50"
        """
        if self.df is None:
            self.load_data()
        
        df_filtered = self.df.copy()
        
        if query_string:
            try:
                df_filtered = df_filtered.query(query_string)
                print(f"✅ Filtro aplicado via query: {query_string}")
            except Exception as e:
                print(f"❌ Erro na query: {e}")
        
        if filter_dict:
            for col, value in filter_dict.items():
                if col in df_filtered.columns:
                    if isinstance(value, list):
                        df_filtered = df_filtered[df_filtered[col].isin(value)]
                    else:
                        df_filtered = df_filtered[df_filtered[col] == value]
                    print(f"✅ Filtro aplicado: {col} = {value}")
        
        self.filtered_df = df_filtered
        self.filters_applied = True
        
        print(f"📊 Após filtros: {len(self.filtered_df):,} registros (de {len(self.df):,})")
        
        return self.filtered_df
    
    # =====================================================
    # 🔥 NOVO MÉTODO: RESET FILTERS
    # =====================================================
    def reset_filters(self):
        """Remove todos os filtros e retorna ao DataFrame original"""
        if self.df is not None:
            self.filtered_df = self.df.copy()
            self.filters_applied = False
            print("✅ Filtros removidos")
            return self.filtered_df
        return None
    
    # =====================================================
    # 🔥 NOVO MÉTODO: GET_FILTERED_DF (Alias para compatibilidade)
    # =====================================================
    def get_filtered_df(self):
        """Alias para get_current_df() - compatibilidade"""
        return self.get_current_df()
    
    # =====================================================
    # 🔥 NOVO MÉTODO: REFRESH DATA
    # =====================================================
    def refresh_data(self, force_reload=False):
        """
        Recarrega os dados do Excel (ignora cache se force_reload=True)
        Mantém os filtros atuais se possível
        """
        old_filters_applied = self.filters_applied
        old_filtered_df = self.filtered_df if self.filters_applied else None
        
        # Recarrega dados
        self.load_data(force_reload=force_reload)
        
        # Tenta reaplicar filtros se existiam
        if old_filters_applied and old_filtered_df is not None:
            # Isso é simplificado - você pode implementar lógica mais específica
            print("⚠️ Filtros anteriores foram resetados após refresh")
            self.filters_applied = False
        
        return self.df

    # =====================================================
    # LOCALIZAR EXCEL (PORTÁTIL)
    # =====================================================
    def _find_excel(self):

        # 📌 base do projeto (não depende do terminal)
        base_dir = Path(__file__).resolve().parent.parent

        arquivos = list(base_dir.rglob("*.xlsx"))

        if not arquivos:
            return None

        # 🔥 pega o mais recente (mais seguro)
        return str(
            max(arquivos, key=lambda f: f.stat().st_mtime)
        )

    # =====================================================
    # VALIDAR CACHE
    # =====================================================
    def _cache_valido(self, excel_file):

        if not self.parquet_path.exists():
            return False

        parquet_time = self.parquet_path.stat().st_mtime
        excel_time = Path(excel_file).stat().st_mtime

        return parquet_time >= excel_time

    # =====================================================
    # SALVAR PARQUET
    # =====================================================
    def _save_cache(self):

        self.parquet_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.df.to_parquet(
            self.parquet_path,
            index=False,
            compression="snappy"
        )

        print("💾 Cache Parquet salvo")

    # =====================================================
    # PREPROCESSAMENTO
    # =====================================================
    def _preprocess(self):

        if "date" in self.df.columns:

            self.df["date"] = pd.to_datetime(
                self.df["date"],
                errors="coerce"
            )

            self.df["month"] = self.df["date"].dt.month
            self.df["year"] = self.df["date"].dt.year

        if "sleep_efficiency" in self.df.columns:

            self.df["sleep_quality"] = pd.cut(
                self.df["sleep_efficiency"],
                bins=[0, 70, 85, 100],
                labels=["Ruim", "Bom", "Excelente"]
            )

        if "recovery_score" in self.df.columns:

            self.df["recovery_level"] = pd.cut(
                self.df["recovery_score"],
                bins=[0, 33, 66, 100],
                labels=["Baixo", "Médio", "Alto"]
            )

        # otimização de memória
        for col in ["gender", "fitness_level", "primary_sport"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype("category")

    # =====================================================
    # LOG
    # =====================================================
    def _log(self, inicio, origem):

        tempo = time.time() - inicio

        print("===================================")
        print(f"✅ Dados carregados via: {origem}")
        print(f"📊 Registros: {len(self.df):,}")
        print(f"📁 Colunas: {len(self.df.columns)}")
        print(f"⏱ Tempo: {tempo:.2f}s")
        print("===================================")


# =====================================================
# INSTÂNCIA GLOBAL (DASH)
# =====================================================
data_manager = DataManager()


# =====================================================
# TESTE LOCAL
# =====================================================
if __name__ == "__main__":

    df = data_manager.load_data()
    
    print("\n" + "="*50)
    print("TESTANDO NOVOS MÉTODOS:")
    print("="*50)
    
    # Testar get_current_df
    current_df = data_manager.get_current_df()
    print(f"✅ get_current_df(): {current_df.shape}")
    
    # Testar filtros (exemplo)
    if 'gender' in current_df.columns:
        filtered = data_manager.apply_filters({'gender': 'Male'})
        print(f"✅ Após filtro de gênero: {filtered.shape}")
        
        # Resetar filtros
        reset_df = data_manager.reset_filters()
        print(f"✅ Após reset: {reset_df.shape}")
    
    print("\n" + "="*50)
    print("✅ DataManager atualizado com sucesso!")
    print("="*50)