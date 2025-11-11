You are an expert Python ML engineer. Generate a clean, production-ready ML core module for the IRIS dataset to be imported by a Django API. Follow these exact requirements.

## Goals
- Provide a small, importable module that:
  1) trains and saves models/metrics artifacts,
  2) loads pre-trained models in memory,
  3) serves predictions for a given model name,
  4) exposes helpers for feature order and results lookup.
- No heavy work on import. Expensive work should be done in explicit functions.

## Files to create
1) ml_core.py  ← main public API
2) models/     ← directory for artifacts (created if missing)
   - rf_model.pkl
   - svc_model.pkl
   - results.json

## Tech/Libs
- Python 3.x
- numpy, pandas (optional), scikit-learn, joblib, shap (only needed later by API XAI, but keep the core compatible)
- Use pathlib for paths and logging for debug information.

## Dataset and feature order
- Use sklearn.datasets.load_iris.
- Fixed feature order (document and enforce on predict):
  ["sepal_length", "sepal_width", "petal_length", "petal_width"]
- Use target_names from the dataset to map predicted class index → label.

## Determinism and split
- train_test_split with test_size=0.20, stratify=y, random_state=42.

## Models to train
- RandomForestClassifier(n_estimators=200, random_state=42)
- SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42 if available)
- Store with joblib as:
  - models/rf_model.pkl
  - models/svc_model.pkl

## Metrics and results.json
- Compute for **test set**: accuracy, precision_macro, recall_macro, f1_macro.
- Keep per-model hyperparameters of interest in results.json.
- Include metadata:
  - dataset: {"name": "iris", "n_features": 4, "n_classes": 3, "feature_order": [...], "target_names": [...]}
  - split: {"test_size": 0.2, "random_state": 42, "n_train": <int>, "n_test": <int>}
  - models: [
      {"name":"rf",  "type":"RandomForestClassifier", "params":{...}, "metrics":{"accuracy":..., "precision_macro":..., "recall_macro":..., "f1_macro":...}},
      {"name":"svc", "type":"SVC",                    "params":{...}, "metrics":{"accuracy":..., "precision_macro":..., "recall_macro":..., "f1_macro":...}}
    ]
  - best_model: select by highest accuracy; tie-breaker: highest f1_macro; if still tie, pick "rf".
- Save to models/results.json (UTF-8, pretty=2).

## Public API (ml_core.py)
Provide type hints, docstrings (Google or NumPy style), and logging. Raise a custom exception MLCoreError on validation/load issues.

- class MLCoreError(Exception): pass

- def get_feature_order() -> list[str]:
    """Return the canonical IRIS feature order."""
    # ["sepal_length", "sepal_width", "petal_length", "petal_width"]

- def train_and_save_models(models_dir: str = "models") -> dict:
    """
    Train RF and SVC on IRIS with a deterministic split, compute metrics,
    persist PKL artifacts and results.json. Return the loaded JSON dict.
    """
    # Create models_dir if missing, train models, evaluate, write artifacts/results.

- def load_models(models_dir: str = "models") -> dict[str, object]:
    """
    Load and return a dict with keys {"rf": estimator, "svc": estimator}.
    Raise MLCoreError if any artifact is missing or corrupted.
    """

- def load_results(results_path: str = "models/results.json") -> dict:
    """Load and return the JSON results with metadata and metrics."""

- def predict_with_model(model_name: str, features: list[float] | tuple[float, float, float, float],
                         models: dict[str, object] | None = None,
                         models_dir: str = "models") -> dict:
    """
    Validate model_name in {"rf","svc"} and features length==4 and numeric.
    Load models if 'models' is None. Return:
    {
      "model_name": "rf",
      "input_feature_order": [...],
      "input_features": [...],
      "prediction_index": <int>,
      "prediction_label": <str>,
      "prediction_proba": [p_setosa, p_versicolor, p_virginica]  # If model supports predict_proba; otherwise use calibrated decision_function→softmax fallback.
    }
    Use target_names to build prediction_label.
    For SVC with probability=True, use predict_proba; for RF, predict_proba is available.
    Ensure probabilities sum ~1.0 (round to, e.g., 6 decimals).
    """

## Input validation
- features must be length 4, all finite numbers. Raise MLCoreError on invalid input and include a helpful message.
- model_name must be "rf" or "svc".

## Code quality
- Use pathlib.Path, not bare strings for FS operations.
- No top-level heavy work; importing ml_core must be fast.
- Add __main__ guard to allow CLI:
    if __name__ == "__main__":
        # trains and prints a compact summary; DO NOT run on import
- Keep functions pure where possible and easy to unit test.

## Logging
- Basic logging setup in this module (INFO default). Log key steps:
  - dataset load, split sizes
  - start/end of training each model
  - metrics per model
  - writing artifacts/results
  - model loading and prediction

## Minimal examples (include in docstrings)
Example to train:
>>> from ml_core import train_and_save_models
>>> results = train_and_save_models()
>>> results["best_model"]

Example to predict:
>>> from ml_core import load_models, predict_with_model, get_feature_order
>>> models = load_models()
>>> order = get_feature_order()
>>> x = [5.1, 3.5, 1.4, 0.2]
>>> out = predict_with_model("rf", x, models=models)
>>> out["prediction_label"]

## Acceptance criteria
- Running train_and_save_models() creates models/ with two PKLs and a results.json matching the schema above.
- load_models() returns both models without side effects.
- predict_with_model(...) returns label and probabilities (list of 3 floats).
- All public functions have type hints and docstrings; module raises MLCoreError on invalid inputs/missing artifacts.

## Extras (nice-to-have)
- Export a constant FEATURE_ORDER at module level.
- Tiny utility _ensure_dir(path: Path) for directory creation.
- Tie-breaking logic well tested and logged.

Generate only the code for ml_core.py (and create the models/ folder at runtime). Do not write tests now. Keep the module self-contained and ready to be imported by a Django view/service.
