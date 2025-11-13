"""
Servicio para cargar y gestionar modelos de ML
"""
import os
import joblib
import json
from pathlib import Path


class ModelService:
    """
    Servicio singleton para cargar y gestionar modelos de ML
    """
    _instance = None
    _models = {}
    _metadata = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.load_models()
    
    def load_models(self):
        """
        Carga todos los modelos PKL desde la carpeta models/
        """
        # Obtener la ruta base del proyecto
        base_dir = Path(__file__).resolve().parent.parent.parent.parent
        models_dir = base_dir / 'models'
        
        print(f"[ModelService] Buscando modelos en: {models_dir}")
        
        if not models_dir.exists():
            print(f"[ModelService] ⚠️  La carpeta {models_dir} no existe")
            return
        
        # Cargar metadata desde results.json
        results_path = models_dir / 'results.json'
        if results_path.exists():
            with open(results_path, 'r') as f:
                self._metadata = json.load(f)
            print(f"[ModelService] ✓ Metadata cargada desde {results_path}")
        else:
            print(f"[ModelService] ⚠️  No se encontró results.json")
        
        # Cargar todos los archivos .pkl
        pkl_files = list(models_dir.glob('*.pkl'))
        
        if not pkl_files:
            print(f"[ModelService] ⚠️  No se encontraron archivos .pkl en {models_dir}")
            return
        
        for pkl_file in pkl_files:
            try:
                model_name = pkl_file.stem  # nombre sin extensión
                model = joblib.load(pkl_file)
                self._models[model_name] = model
                print(f"[ModelService] ✓ Modelo '{model_name}' cargado desde {pkl_file.name}")
            except Exception as e:
                print(f"[ModelService] ✗ Error al cargar {pkl_file.name}: {e}")
        
        print(f"[ModelService] Total de modelos cargados: {len(self._models)}")
    
    def get_model(self, model_name):
        """
        Obtiene un modelo por su nombre
        
        Args:
            model_name: Nombre del modelo (ej: 'rf_model' o 'rf')
        
        Returns:
            El modelo cargado o None si no existe
        """
        # Intentar con el nombre exacto
        if model_name in self._models:
            return self._models[model_name]
        
        # Intentar agregando '_model'
        model_name_with_suffix = f"{model_name}_model"
        if model_name_with_suffix in self._models:
            return self._models[model_name_with_suffix]
        
        return None
    
    def get_all_models(self):
        """
        Retorna todos los modelos cargados
        """
        return self._models
    
    def get_metadata(self):
        """
        Retorna la metadata de los modelos desde results.json
        """
        return self._metadata
    
    def get_available_models(self):
        """
        Retorna una lista con los nombres de los modelos disponibles
        """
        return list(self._models.keys())


# Instancia global del servicio
model_service = ModelService()
