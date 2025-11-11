"""
Script de entrenamiento de modelos de clasificación usando el dataset IRIS.

Este script entrena múltiples clasificadores (RandomForest y SVC) sobre el dataset IRIS,
evalúa su rendimiento en un conjunto de prueba, guarda los modelos entrenados en archivos
PKL y registra todas las métricas y metadatos en un archivo JSON para su posterior consumo
desde una API.

Autor: Generado para proyecto IRIS ML Deployment
Fecha: 2025-11-06
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


def get_data(test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Carga el dataset IRIS y lo divide en conjuntos de entrenamiento y prueba.
    
    Args:
        test_size: Proporción del dataset para el conjunto de prueba.
        random_state: Semilla para reproducibilidad.
    
    Returns:
        Tupla con (X_train, X_test, y_train, y_test, metadata)
    """
    print("Cargando dataset IRIS...")
    
    # Cargar dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Metadata del dataset
    metadata = {
        "name": "iris",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_names": iris.feature_names,
        "target_names": iris.target_names.tolist()
    }
    
    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"Dataset cargado: {metadata['n_samples']} muestras, {metadata['n_features']} features")
    print(f"Train: {len(X_train)} muestras | Test: {len(X_test)} muestras")
    
    return X_train, X_test, y_train, y_test, metadata


def train_models(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    """
    Entrena múltiples modelos de clasificación.
    
    Args:
        X_train: Features de entrenamiento.
        y_train: Etiquetas de entrenamiento.
    
    Returns:
        Diccionario con los modelos entrenados y sus parámetros.
    """
    models_config = {
        "rf": {
            "model": RandomForestClassifier(n_estimators=100, random_state=42),
            "type": "RandomForestClassifier",
            "params": {
                "n_estimators": 100,
                "random_state": 42
            }
        },
        "svc": {
            "model": SVC(kernel="rbf", probability=True, random_state=42),
            "type": "SVC",
            "params": {
                "kernel": "rbf",
                "probability": True,
                "random_state": 42
            }
        }
    }
    
    trained_models = {}
    
    for model_id, config in models_config.items():
        print(f"\nEntrenando modelo {model_id} ({config['type']})...")
        model = config["model"]
        model.fit(X_train, y_train)
        
        trained_models[model_id] = {
            "model": model,
            "type": config["type"],
            "params": config["params"]
        }
        print(f"Modelo {model_id} entrenado exitosamente")
    
    return trained_models


def evaluate_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Evalúa los modelos entrenados en el conjunto de prueba.
    
    Args:
        models: Diccionario con los modelos entrenados.
        X_test: Features de prueba.
        y_test: Etiquetas de prueba.
    
    Returns:
        Diccionario con las métricas de cada modelo.
    """
    results = {}
    
    for model_id, model_info in models.items():
        print(f"\nEvaluando modelo {model_id}...")
        model = model_info["model"]
        
        # Predicciones
        y_pred = model.predict(X_test)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        
        results[model_id] = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }
        
        print(f"Accuracy {model_id}: {accuracy:.4f}")
        print(f"Precision {model_id}: {precision:.4f}")
        print(f"Recall {model_id}: {recall:.4f}")
        print(f"F1-Score {model_id}: {f1:.4f}")
    
    return results


def save_models_and_results(
    models: Dict[str, Any],
    metrics: Dict[str, Dict[str, float]],
    metadata: Dict[str, Any],
    split_info: Dict[str, Any],
    models_dir: str = "models"
) -> None:
    """
    Guarda los modelos entrenados en archivos PKL y los resultados en JSON.
    
    Args:
        models: Diccionario con los modelos entrenados.
        metrics: Métricas de evaluación de cada modelo.
        metadata: Metadata del dataset.
        split_info: Información sobre la partición de datos.
        models_dir: Directorio donde guardar los archivos.
    """
    # Crear directorio si no existe
    models_path = Path(models_dir)
    models_path.mkdir(exist_ok=True)
    
    print(f"\nGuardando modelos en la carpeta {models_dir}/...")
    
    # Guardar cada modelo como PKL
    for model_id, model_info in models.items():
        model_filename = models_path / f"{model_id}_model.pkl"
        joblib.dump(model_info["model"], model_filename)
        print(f"Modelo guardado: {model_filename}")
    
    # Determinar el mejor modelo basado en accuracy
    best_model_id = max(metrics.items(), key=lambda x: x[1]["accuracy"])[0]
    
    # Construir estructura de resultados
    results = {
        "dataset": metadata,
        "split": split_info,
        "models": [
            {
                "id": model_id,
                "type": model_info["type"],
                "params": model_info["params"],
                "metrics": metrics[model_id]
            }
            for model_id, model_info in models.items()
        ],
        "best_model": {
            "id": best_model_id,
            "criterion": "accuracy"
        }
    }
    
    # Guardar resultados en JSON
    results_filename = models_path / "results.json"
    with open(results_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados guardados en: {results_filename}")
    print(f"Mejor modelo: {best_model_id} (accuracy: {metrics[best_model_id]['accuracy']:.4f})")


def main() -> None:
    """
    Función principal que orquesta el proceso de entrenamiento.
    """
    print("=" * 60)
    print("ENTRENAMIENTO DE MODELOS - DATASET IRIS")
    print("=" * 60)
    
    # Configuración
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    MODELS_DIR = "models"
    
    # 1. Cargar y preparar datos
    X_train, X_test, y_train, y_test, metadata = get_data(
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    
    # Información del split
    split_info = {
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test))
    }
    
    # 2. Entrenar modelos
    trained_models = train_models(X_train, y_train)
    
    # 3. Evaluar modelos
    metrics = evaluate_models(trained_models, X_test, y_test)
    
    # 4. Guardar modelos y resultados
    save_models_and_results(
        models=trained_models,
        metrics=metrics,
        metadata=metadata,
        split_info=split_info,
        models_dir=MODELS_DIR
    )
    
    print("\n" + "=" * 60)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("=" * 60)


if __name__ == "__main__":
    main()
