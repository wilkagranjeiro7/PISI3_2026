# model_manager.py
import joblib
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

class ModelManager:
    def __init__(self, model_dir='saved_models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
    def salvar_modelo(self, model, scaler, features, metrics, nome_modelo=None):
        """Salva o modelo, scaler e metadados"""
        if nome_modelo is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nome_modelo = f"best_model_{timestamp}"
        
        # Salvar modelo
        model_path = os.path.join(self.model_dir, f"{nome_modelo}.pkl")
        joblib.dump(model, model_path)
        
        # Salvar scaler
        if scaler is not None:
            scaler_path = os.path.join(self.model_dir, f"{nome_modelo}_scaler.pkl")
            joblib.dump(scaler, scaler_path)
        
        # Salvar metadados
        metadata = {
            'nome': nome_modelo,
            'data_criacao': datetime.now().isoformat(),
            'features': features,
            'metricas': metrics
        }
        
        metadata_path = os.path.join(self.model_dir, f"{nome_modelo}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Salvar como "melhor modelo" atual
        self._salvar_modelo_atual(nome_modelo)
        
        return nome_modelo
    
    def _salvar_modelo_atual(self, nome_modelo):
        current_path = os.path.join(self.model_dir, 'current_model.json')
        with open(current_path, 'w') as f:
            json.dump({'nome': nome_modelo}, f)
    
    def carregar_melhor_modelo(self):
        current_path = os.path.join(self.model_dir, 'current_model.json')
        
        if not os.path.exists(current_path):
            return None, None, None, None
        
        with open(current_path, 'r') as f:
            data = json.load(f)
            nome_modelo = data['nome']
        
        model_path = os.path.join(self.model_dir, f"{nome_modelo}.pkl")
        scaler_path = os.path.join(self.model_dir, f"{nome_modelo}_scaler.pkl")
        metadata_path = os.path.join(self.model_dir, f"{nome_modelo}_metadata.json")
        
        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            return None, None, None, None
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return model, scaler, metadata['features'], metadata

# Instância global
model_manager = ModelManager()