"""
ML Core Module for IRIS Classification

This module provides a production-ready API for training, loading, and predicting
with machine learning models on the IRIS dataset. Designed to be imported by
Django API services without performing heavy work on import.

Public API:
    - MLCoreError: Custom exception for validation and artifact errors
    - FEATURE_ORDER: Canonical order of IRIS features
    - get_feature_order(): Returns the feature order as a list
    - train_and_save_models(): Trains models and saves artifacts
    - load_models(): Loads pre-trained models from disk
    - load_results(): Loads metrics and metadata from results.json
    - predict_with_model(): Makes predictions with a specified model

Example usage:
    >>> from ml_core import train_and_save_models, load_models, predict_with_model
    >>> # Train and save
    >>> results = train_and_save_models()
    >>> print(results["best_model"])
    >>> 
    >>> # Load and predict
    >>> models = load_models()
    >>> prediction = predict_with_model("rf", [5.1, 3.5, 1.4, 0.2], models=models)
    >>> print(prediction["prediction_label"])

Author: Generated for IRIS ML Deployment
Date: 2025-11-11
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


# Module-level constants
FEATURE_ORDER = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLCoreError(Exception):
    """Custom exception for ML core validation and artifact errors."""
    pass


def _ensure_dir(path: Path) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    path.mkdir(parents=True, exist_ok=True)


def _validate_features(features: Union[List[float], Tuple[float, ...]]) -> List[float]:
    """
    Validate that features are a list/tuple of 4 finite numbers.
    
    Args:
        features: Input features to validate
        
    Returns:
        Validated features as a list
        
    Raises:
        MLCoreError: If validation fails
    """
    if not isinstance(features, (list, tuple)):
        raise MLCoreError(
            f"Features must be a list or tuple, got {type(features).__name__}"
        )
    
    if len(features) != 4:
        raise MLCoreError(
            f"Features must have exactly 4 values (got {len(features)}). "
            f"Expected order: {FEATURE_ORDER}"
        )
    
    validated = []
    for i, val in enumerate(features):
        try:
            num_val = float(val)
            if not np.isfinite(num_val):
                raise MLCoreError(
                    f"Feature at index {i} ({FEATURE_ORDER[i]}) is not finite: {val}"
                )
            validated.append(num_val)
        except (TypeError, ValueError) as e:
            raise MLCoreError(
                f"Feature at index {i} ({FEATURE_ORDER[i]}) is not a valid number: {val}"
            ) from e
    
    return validated


def _validate_model_name(model_name: str) -> None:
    """
    Validate that model_name is one of the supported models.
    
    Args:
        model_name: Model identifier
        
    Raises:
        MLCoreError: If model name is invalid
    """
    valid_names = {"rf", "svc"}
    if model_name not in valid_names:
        raise MLCoreError(
            f"Invalid model_name '{model_name}'. Must be one of: {valid_names}"
        )


def get_feature_order() -> List[str]:
    """
    Return the canonical IRIS feature order.
    
    Returns:
        List of feature names in the expected order
        
    Example:
        >>> order = get_feature_order()
        >>> print(order)
        ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    """
    return FEATURE_ORDER.copy()


def train_and_save_models(models_dir: str = "models") -> Dict[str, Any]:
    """
    Train RandomForest and SVC on IRIS with deterministic split, compute metrics,
    and persist PKL artifacts and results.json.
    
    This function:
    1. Loads the IRIS dataset
    2. Splits data (80/20, stratified, random_state=42)
    3. Trains RandomForest and SVC classifiers
    4. Evaluates on test set (accuracy, precision, recall, F1)
    5. Saves models as PKL files
    6. Saves metadata and metrics to results.json
    
    Args:
        models_dir: Directory where artifacts will be saved (default: "models")
        
    Returns:
        Dictionary containing metadata, metrics, and best model info
        
    Raises:
        MLCoreError: If training or saving fails
        
    Example:
        >>> results = train_and_save_models()
        >>> print(f"Best model: {results['best_model']['name']}")
        >>> print(f"Accuracy: {results['models'][0]['metrics']['accuracy']:.4f}")
    """
    logger.info("Starting model training pipeline")
    
    # Create models directory
    models_path = Path(models_dir)
    _ensure_dir(models_path)
    
    # Load IRIS dataset
    logger.info("Loading IRIS dataset")
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names.tolist()
    
    logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    
    n_train = len(X_train)
    n_test = len(X_test)
    logger.info(f"Data split: train={n_train}, test={n_test}")
    
    # Define models
    models_config = {
        "rf": {
            "estimator": RandomForestClassifier(n_estimators=200, random_state=42),
            "type": "RandomForestClassifier",
            "params": {
                "n_estimators": 200,
                "random_state": 42
            }
        },
        "svc": {
            "estimator": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42),
            "type": "SVC",
            "params": {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
                "probability": True,
                "random_state": 42
            }
        }
    }
    
    # Train and evaluate models
    trained_models = {}
    models_results = []
    
    for model_name, config in models_config.items():
        logger.info(f"Training model: {model_name} ({config['type']})")
        
        # Train
        estimator = config["estimator"]
        estimator.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = estimator.predict(X_test)
        
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro')
        recall_macro = recall_score(y_test, y_pred, average='macro')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        logger.info(
            f"Model {model_name} - Accuracy: {accuracy:.4f}, "
            f"Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, "
            f"F1: {f1_macro:.4f}"
        )
        
        # Save model
        model_path = models_path / f"{model_name}_model.pkl"
        joblib.dump(estimator, model_path)
        logger.info(f"Saved model to: {model_path}")
        
        # Store results
        trained_models[model_name] = estimator
        models_results.append({
            "name": model_name,
            "type": config["type"],
            "params": config["params"],
            "metrics": {
                "accuracy": float(accuracy),
                "precision_macro": float(precision_macro),
                "recall_macro": float(recall_macro),
                "f1_macro": float(f1_macro)
            }
        })
    
    # Determine best model (highest accuracy; tie-breaker: highest f1_macro; second tie: "rf")
    best_model_info = max(
        models_results,
        key=lambda m: (m["metrics"]["accuracy"], m["metrics"]["f1_macro"], m["name"] == "rf")
    )
    best_model_name = best_model_info["name"]
    
    logger.info(
        f"Best model: {best_model_name} "
        f"(accuracy={best_model_info['metrics']['accuracy']:.4f})"
    )
    
    # Build results structure
    results = {
        "dataset": {
            "name": "iris",
            "n_features": int(X.shape[1]),
            "n_classes": int(len(target_names)),
            "feature_order": FEATURE_ORDER.copy(),
            "target_names": target_names
        },
        "split": {
            "test_size": 0.2,
            "random_state": 42,
            "n_train": n_train,
            "n_test": n_test
        },
        "models": models_results,
        "best_model": {
            "name": best_model_name,
            "criterion": "accuracy (tie-breaker: f1_macro)"
        }
    }
    
    # Save results.json
    results_path = models_path / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved results to: {results_path}")
    logger.info("Training pipeline completed successfully")
    
    return results


def load_models(models_dir: str = "models") -> Dict[str, Any]:
    """
    Load pre-trained models from disk.
    
    Args:
        models_dir: Directory containing the PKL files (default: "models")
        
    Returns:
        Dictionary with keys {"rf": estimator, "svc": estimator}
        
    Raises:
        MLCoreError: If any artifact is missing or corrupted
        
    Example:
        >>> models = load_models()
        >>> print(list(models.keys()))
        ['rf', 'svc']
    """
    logger.info(f"Loading models from: {models_dir}")
    
    models_path = Path(models_dir)
    
    if not models_path.exists():
        raise MLCoreError(
            f"Models directory does not exist: {models_path}. "
            "Run train_and_save_models() first."
        )
    
    loaded_models = {}
    
    for model_name in ["rf", "svc"]:
        model_file = models_path / f"{model_name}_model.pkl"
        
        if not model_file.exists():
            raise MLCoreError(
                f"Model file not found: {model_file}. "
                "Run train_and_save_models() first."
            )
        
        try:
            loaded_models[model_name] = joblib.load(model_file)
            logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            raise MLCoreError(
                f"Failed to load model from {model_file}: {e}"
            ) from e
    
    logger.info("All models loaded successfully")
    return loaded_models


def load_results(results_path: str = "models/results.json") -> Dict[str, Any]:
    """
    Load and return the JSON results with metadata and metrics.
    
    Args:
        results_path: Path to results.json (default: "models/results.json")
        
    Returns:
        Dictionary containing dataset info, split info, model metrics, and best model
        
    Raises:
        MLCoreError: If results file is missing or corrupted
        
    Example:
        >>> results = load_results()
        >>> print(results["best_model"]["name"])
        >>> print(results["dataset"]["feature_order"])
    """
    logger.info(f"Loading results from: {results_path}")
    
    results_file = Path(results_path)
    
    if not results_file.exists():
        raise MLCoreError(
            f"Results file not found: {results_file}. "
            "Run train_and_save_models() first."
        )
    
    try:
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        logger.info("Results loaded successfully")
        return results
    except Exception as e:
        raise MLCoreError(
            f"Failed to load results from {results_file}: {e}"
        ) from e


def predict_with_model(
    model_name: str,
    features: Union[List[float], Tuple[float, ...]],
    models: Optional[Dict[str, Any]] = None,
    models_dir: str = "models"
) -> Dict[str, Any]:
    """
    Make a prediction using a specified model.
    
    This function:
    1. Validates model_name and features
    2. Loads models if not provided
    3. Makes prediction and computes probabilities
    4. Returns structured prediction result
    
    Args:
        model_name: Model identifier ("rf" or "svc")
        features: List or tuple of 4 numeric values in the order:
                 [sepal_length, sepal_width, petal_length, petal_width]
        models: Pre-loaded models dict (optional, will load if None)
        models_dir: Directory containing models (used if models is None)
        
    Returns:
        Dictionary containing:
            - model_name: Name of the model used
            - input_feature_order: Feature names in expected order
            - input_features: Validated input features
            - prediction_index: Predicted class index (0, 1, or 2)
            - prediction_label: Predicted class name
            - prediction_proba: List of 3 probabilities for each class
            
    Raises:
        MLCoreError: If validation fails or prediction error occurs
        
    Example:
        >>> models = load_models()
        >>> features = [5.1, 3.5, 1.4, 0.2]
        >>> result = predict_with_model("rf", features, models=models)
        >>> print(f"Predicted: {result['prediction_label']}")
        >>> print(f"Probabilities: {result['prediction_proba']}")
    """
    # Validate inputs
    _validate_model_name(model_name)
    validated_features = _validate_features(features)
    
    logger.info(f"Making prediction with model: {model_name}")
    logger.debug(f"Input features: {validated_features}")
    
    # Load models if not provided
    if models is None:
        models = load_models(models_dir=models_dir)
    
    # Get model
    model = models.get(model_name)
    if model is None:
        raise MLCoreError(
            f"Model '{model_name}' not found in loaded models. "
            f"Available: {list(models.keys())}"
        )
    
    # Load results to get target names
    results = load_results(results_path=str(Path(models_dir) / "results.json"))
    target_names = results["dataset"]["target_names"]
    
    # Prepare input
    X_input = np.array(validated_features).reshape(1, -1)
    
    # Make prediction
    try:
        pred_index = int(model.predict(X_input)[0])
        pred_label = target_names[pred_index]
        
        # Get probabilities
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)[0]
        else:
            # Fallback for models without predict_proba
            # (shouldn't happen with our SVC since probability=True)
            logger.warning(f"Model {model_name} doesn't support predict_proba")
            proba = np.zeros(len(target_names))
            proba[pred_index] = 1.0
        
        # Round probabilities to 6 decimals
        proba_list = [round(float(p), 6) for p in proba]
        
        logger.info(f"Prediction: {pred_label} (index={pred_index})")
        logger.debug(f"Probabilities: {proba_list}")
        
        return {
            "model_name": model_name,
            "input_feature_order": FEATURE_ORDER.copy(),
            "input_features": validated_features,
            "prediction_index": pred_index,
            "prediction_label": pred_label,
            "prediction_proba": proba_list
        }
        
    except Exception as e:
        raise MLCoreError(
            f"Prediction failed for model '{model_name}': {e}"
        ) from e


def main() -> None:
    """
    CLI entry point for training models and displaying summary.
    
    This function is called when the module is run as a script.
    It trains models, saves artifacts, and prints a summary.
    """
    print("=" * 70)
    print("ML CORE - IRIS Classification Model Training")
    print("=" * 70)
    
    # Train and save
    results = train_and_save_models()
    
    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    
    print(f"\nDataset: {results['dataset']['name'].upper()}")
    print(f"Features: {results['dataset']['n_features']}")
    print(f"Classes: {results['dataset']['n_classes']}")
    print(f"Feature order: {', '.join(results['dataset']['feature_order'])}")
    
    print(f"\nSplit: {results['split']['n_train']} train / {results['split']['n_test']} test")
    print(f"Test size: {results['split']['test_size']}")
    print(f"Random state: {results['split']['random_state']}")
    
    print("\nModel Performance:")
    for model_info in results['models']:
        print(f"\n  {model_info['name'].upper()} ({model_info['type']})")
        metrics = model_info['metrics']
        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision_macro']:.4f}")
        print(f"    Recall:    {metrics['recall_macro']:.4f}")
        print(f"    F1-Score:  {metrics['f1_macro']:.4f}")
    
    print(f"\nBest Model: {results['best_model']['name'].upper()}")
    print(f"Criterion: {results['best_model']['criterion']}")
    
    print("\n" + "=" * 70)
    print("ARTIFACTS SAVED")
    print("=" * 70)
    print("  - models/rf_model.pkl")
    print("  - models/svc_model.pkl")
    print("  - models/results.json")
    
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTION")
    print("=" * 70)
    
    # Demo prediction
    models = load_models()
    test_features = [5.1, 3.5, 1.4, 0.2]
    pred = predict_with_model("rf", test_features, models=models)
    
    print(f"\nInput: {test_features}")
    print(f"Feature order: {pred['input_feature_order']}")
    print(f"Predicted class: {pred['prediction_label']}")
    print(f"Probabilities: {pred['prediction_proba']}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
