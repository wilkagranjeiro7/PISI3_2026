# ==================================================
# model_manager.py - GERENCIADOR DE MODELOS (COMPLETO)
# ==================================================

import pickle
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

class ModelManager:
    """Gerencia o salvamento e carregamento de modelos"""
    
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
    
    # ================================================
    # SALVAR MODELO DE CLASSIFICAÇÃO
    # ================================================
    def salvar_modelo_classificacao(self, model, scaler, features, metricas, df_referencia=None):
        """
        Salva o modelo de classificação com metadata
        
        Parâmetros:
        - model: modelo treinado
        - scaler: objeto StandardScaler (ou None)
        - features: lista de features usadas
        - metricas: dicionário com métricas do modelo
        - df_referencia: DataFrame para calcular referências
        """
        
        metadata = {
            'tipo': 'classificacao',
            'features': features,
            'metricas': metricas,
            'data_criacao': datetime.now().isoformat(),
            'nome': f"Classificador_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        # Calcular referências se fornecido
        if df_referencia is not None:
            metadata['referencias'] = self._calcular_referencias(df_referencia, features)
        
        # Salvar
        caminho = self.model_dir / "modelo_classificacao.pkl"
        with open(caminho, 'wb') as f:
            pickle.dump({
                'model': model,
                'scaler': scaler,
                'features': features,
                'metadata': metadata
            }, f)
        
        print(f"✅ Modelo de classificação salvo em: {caminho}")
        return str(caminho)
    
    # ================================================
    # SALVAR MODELO DE REGRESSÃO
    # ================================================
    def salvar_modelo_regressao(self, model, scaler, features, metricas, df_referencia=None):
        """
        Salva o modelo de regressão com metadata
        """
        
        metadata = {
            'tipo': 'regressao',
            'features': features,
            'metricas': metricas,
            'data_criacao': datetime.now().isoformat(),
            'nome': f"Regressor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        if df_referencia is not None:
            metadata['referencias'] = self._calcular_referencias(df_referencia, features)
        
        caminho = self.model_dir / "modelo_regressao.pkl"
        with open(caminho, 'wb') as f:
            pickle.dump({
                'model': model,
                'scaler': scaler,
                'features': features,
                'metadata': metadata
            }, f)
        
        print(f"✅ Modelo de regressão salvo em: {caminho}")
        return str(caminho)
    
    # ================================================
    # CARREGAR MODELO DE CLASSIFICAÇÃO
    # ================================================
    def carregar_modelo_classificacao(self):
        """Carrega o modelo de classificação salvo"""
        caminho = self.model_dir / "modelo_classificacao.pkl"
        if not caminho.exists():
            return None, None, None, None
        
        try:
            with open(caminho, 'rb') as f:
                data = pickle.load(f)
            return data['model'], data.get('scaler'), data['features'], data['metadata']
        except Exception as e:
            print(f"❌ Erro ao carregar classificação: {e}")
            return None, None, None, None
    
    # ================================================
    # CARREGAR MODELO DE REGRESSÃO
    # ================================================
    def carregar_modelo_regressao(self):
        """Carrega o modelo de regressão salvo"""
        caminho = self.model_dir / "modelo_regressao.pkl"
        if not caminho.exists():
            return None, None, None, None
        
        try:
            with open(caminho, 'rb') as f:
                data = pickle.load(f)
            return data['model'], data.get('scaler'), data['features'], data['metadata']
        except Exception as e:
            print(f"❌ Erro ao carregar regressão: {e}")
            return None, None, None, None
    
    # ================================================
    # CARREGAR AMBOS OS MODELOS PARA COMPARAÇÃO
    # ================================================
    def carregar_modelos_comparativos(self):
        """
        Carrega ambos os modelos (classificação e regressão) para comparação.
        Retorna um dicionário com os dois modelos.
        """
        model_clf, scaler_clf, features_clf, metadata_clf = self.carregar_modelo_classificacao()
        model_reg, scaler_reg, features_reg, metadata_reg = self.carregar_modelo_regressao()
        
        return {
            'classificacao': {
                'model': model_clf,
                'scaler': scaler_clf,
                'features': features_clf,
                'metadata': metadata_clf
            },
            'regressao': {
                'model': model_reg,
                'scaler': scaler_reg,
                'features': features_reg,
                'metadata': metadata_reg
            }
        }
    
    # ================================================
    # CALCULAR REFERÊNCIAS
    # ================================================
    def _calcular_referencias(self, df, features):
        """Calcula referências para a tela de Insights"""
        referencias = {}
        
        # Percentis do dataset
        percentis = {}
        for feat in features:
            if feat in df.columns:
                dados = df[feat].dropna()
                if len(dados) > 0:
                    percentis[feat] = {
                        'p25': float(dados.quantile(0.25)),
                        'p50': float(dados.quantile(0.50)),
                        'p75': float(dados.quantile(0.75))
                    }
        referencias['percentis'] = percentis
        
        # Médias do grupo de alta recuperação (recovery > 75)
        if 'recovery_score' in df.columns:
            alta_rec = df[df['recovery_score'] > 75]
            medias_alta = {}
            for feat in features:
                if feat in alta_rec.columns:
                    medias_alta[feat] = float(alta_rec[feat].mean())
            referencias['alta_recuperacao'] = medias_alta
        
        # Médias do grupo de baixa recuperação (recovery < 50)
        if 'recovery_score' in df.columns:
            baixa_rec = df[df['recovery_score'] < 50]
            medias_baixa = {}
            for feat in features:
                if feat in baixa_rec.columns:
                    medias_baixa[feat] = float(baixa_rec[feat].mean())
            referencias['baixa_recuperacao'] = medias_baixa
        
        return referencias
    
    # ================================================
    # MÉTODOS AUXILIARES
    # ================================================
    def listar_modelos(self):
        """Lista todos os modelos salvos"""
        modelos = []
        for arquivo in self.model_dir.glob("*.pkl"):
            modelos.append({
                'nome': arquivo.stem,
                'caminho': str(arquivo),
                'tamanho': arquivo.stat().st_size,
                'modificado': datetime.fromtimestamp(arquivo.stat().st_mtime).isoformat()
            })
        return modelos
    
    def deletar_modelo(self, nome):
        """Deleta um modelo específico"""
        caminho = self.model_dir / f"{nome}.pkl"
        if caminho.exists():
            caminho.unlink()
            return True
        return False
    
    def carregar_melhor_modelo(self):
        """Carrega o melhor modelo (para compatibilidade com insights.py)"""
        caminho = self.model_dir / "modelo_classificacao.pkl"
        if not caminho.exists():
            return None, None, None, None
        try:
            with open(caminho, 'rb') as f:
                data = pickle.load(f)
            return data['model'], data.get('scaler'), data['features'], data['metadata']
        except Exception as e:
            print(f"❌ Erro ao carregar melhor modelo: {e}")
            return None, None, None, None

# Instância global
model_manager = ModelManager()