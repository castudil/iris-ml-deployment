# Prompt Copilot – train_models.py (IRIS Deployment)

Fecha: 2025-11-06  
Descripción: Prompt usado para generar el script `train_models.py` (entrenamiento de modelos IRIS, guardado de PKL y JSON de resultados).

Actúa como un desarrollador senior de Python y Machine Learning.

Quiero que generes un script completo llamado `train_models.py` para un proyecto de despliegue de modelos de ML usando el dataset IRIS. El objetivo del script es entrenar varios clasificadores, guardar los modelos entrenados en disco y registrar las métricas en un archivo JSON con todos los detalles necesarios para luego consumirlos desde una API en Django.

Requisitos detallados:

1. Estructura general
   - El script debe poder ejecutarse desde la línea de comando, por ejemplo:
     `python train_models.py`
   - Usa el patrón `if __name__ == "__main__":` para el punto de entrada.
   - Agrega un docstring inicial explicando qué hace el script.
   - No uses notebooks ni dependencias raras; sólo:
     - numpy
     - pandas (si lo necesitas)
     - scikit-learn
     - joblib
     - json
     - os
     - pathlib
     - typing (opcional)

2. Datos y partición
   - Usa el dataset IRIS de `sklearn.datasets.load_iris`.
   - Separa en features X y etiquetas y.
   - Usa `train_test_split` con:
     - `test_size = 0.2` (20% para test)
     - `random_state = 42`
     - `stratify = y` si corresponde.
   - Guarda en variables el tamaño de cada conjunto:
     - `n_train`
     - `n_test`
   - Registra también:
     - nombres de las features
     - nombres de las clases

3. Modelos a entrenar
   - Entrena al menos dos modelos:
     - `RandomForestClassifier` (por ejemplo con `n_estimators=100`, `random_state=42`)
     - `SVC` (por ejemplo con `kernel="rbf"`, `probability=True`, `random_state=42` si aplica)
   - Coloca los modelos en una estructura tipo diccionario, por ejemplo:
     ```python
     models = {
         "rf": RandomForestClassifier(...),
         "svc": SVC(...)
     }
     ```

4. Entrenamiento y evaluación
   - Para cada modelo:
     - Entrénalo con el conjunto de entrenamiento.
     - Evalúalo en el conjunto de test.
     - Calcula al menos:
       - `accuracy` sobre el conjunto de test.
     - Opcional: puedes calcular también precision, recall y f1-score macro.
   - Guarda los resultados en una estructura de Python que luego se convertirá a JSON. Ejemplo de estructura (puedes mejorarla, pero mantén estas ideas):
     ```python
     results = {
         "dataset": {
             "name": "iris",
             "n_samples": int,
             "n_features": int,
             "feature_names": [...],
             "target_names": [...]
         },
         "split": {
             "test_size": 0.2,
             "random_state": 42,
             "n_train": int,
             "n_test": int
         },
         "models": [
             {
                 "id": "rf",
                 "type": "RandomForestClassifier",
                 "params": { ... hiperparámetros usados ... },
                 "metrics": {
                     "accuracy": float
                 }
             },
             {
                 "id": "svc",
                 "type": "SVC",
                 "params": { ... },
                 "metrics": {
                     "accuracy": float
                 }
             }
         ],
         "best_model": {
             "id": "rf",    # el id del mejor modelo según accuracy
             "criterion": "accuracy"
         }
     }
     ```

5. Guardado de modelos (PKL)
   - Crea una carpeta `models` en el mismo nivel del script si no existe (por ejemplo usando `Path("models").mkdir(exist_ok=True)`).
   - Para cada modelo entrenado, guarda un archivo `.pkl` con `joblib.dump`, por ejemplo:
     - `models/rf_model.pkl`
     - `models/svc_model.pkl`

6. Guardado del JSON de resultados
   - Guarda el diccionario `results` en un archivo JSON, por ejemplo:
     - `models/results.json`
   - Asegúrate de usar `indent=2` para que el archivo sea legible.
   - Maneja la serialización de tipos de NumPy si es necesario (por ejemplo convirtiendo a `float(result)`).

7. Mensajes de log en consola
   - Imprime mensajes claros en consola durante la ejecución, por ejemplo:
     - “Cargando dataset IRIS…”
     - “Entrenando modelo rf…”
     - “Accuracy rf: …”
     - “Guardando modelos en la carpeta models/…”
     - “Guardando resultados en models/results.json…”
   - No uses logging avanzado, con `print()` es suficiente.

8. Calidad del código
   - Usa funciones auxiliares claras, por ejemplo:
     - `def get_data(...):`
     - `def train_models(...):`
     - `def evaluate_models(...):`
     - `def save_models_and_results(...):`
   - Usa type hints donde tenga sentido.
   - Código limpio, legible y modular.

Devuélveme el contenido completo del archivo `train_models.py` cumpliendo estos requisitos, listo para copiar/pegar y ejecutar.
