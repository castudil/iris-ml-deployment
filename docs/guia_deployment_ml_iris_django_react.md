# Guía para rehacer el proyecto de Deployment ML (IRIS + Django + React/Vite)

## 0. Visión general del flujo

1. **Core de ML en Python**  
   - Entrenar varios modelos de clasificación (IRIS) y guardar:
     - Modelos en formato **PKL**  
     - Un **JSON** con métricas y metadatos (semillas, tamaños de sets, etc.)
2. **Backend en Django (API)**  
   - Cargar los PKL  
   - Exponer endpoints para:
     - Predicción con un modelo dado  
     - Comparación de todos los modelos  
     - XAI con **SHAP** (imagen + explicación)
3. **Pruebas con Postman**  
   - Probar cada endpoint antes de consumirlo desde el frontend.
4. **Frontend con React + Vite + TailwindCSS**  
   - Formularios para enviar datos a la API  
   - Vistas para mostrar predicciones, comparación de modelos y resultados de XAI

Transversal **Control de versiones en GitHub**  
   - Commits pequeños por etapa para poder hacer rollback sin dolor.

---

## 1. Preparación del entorno

### 1.1. Herramientas necesarias

- **Python 3.x**
- **VS Code** (con Git y GitHub configurados)
- **Node.js** (para Vite/React)
- **Postman** (cliente para probar API)
- **Cuenta GitHub**
- **Copilot en VS Code** (versión pagada habilitada)

### 1.2. Crear repositorio y entorno

1. Crear una carpeta del proyecto, por ejemplo: `iris-deployment-ml`.
2. Inicializar **git**:
   ```bash
   git init
   ```
3. Crear y activar entorno virtual de Python:
   ```bash
   python -m venv .venv
   # Activar (Windows / Mac / Linux según corresponda)
   ```
4. Instalar dependencias básicas de ML:
   ```bash
   pip install numpy pandas scikit-learn shap matplotlib joblib
   ```
5. Crear `.gitignore` con al menos:
   - `.venv/`
   - `__pycache__/`
   - `node_modules/`
   - `*.pkl`
   - `*.pyc`

> **Checkpoint Git:**  
> Commit inicial: `chore: estructura base del proyecto y entorno python`.

---

## 2. Etapa ML: script de entrenamiento y comparación de modelos

### 2.1. Script principal de ML

Archivo sugerido: `model.py` (o `train_models.py`).

Responsabilidades:

1. **Cargar datos IRIS** (desde `sklearn.datasets`).
2. **Dividir en train/test** con una **semilla fija** (`random_state`) y guardar:
   - Tamaño del set de entrenamiento  
   - Tamaño del set de test  
   - Proporción (test_size)
3. **Definir y entrenar modelos**, al menos:
   - `RandomForestClassifier`
   - `SVC`
4. **Calcular métricas** para cada modelo:
   - Accuracy (y opcionalmente precision, recall, F1)
5. **Comparar modelos**:
   - Determinar el “modelo ganador” según una métrica (ej. accuracy).
6. **Guardar artefactos**:
   - Cada modelo en un archivo `PKL` (ej. `rf_model.pkl`, `svc_model.pkl`).
   - Un archivo `results.json` con:
     - Nombre de cada modelo
     - Métricas de desempeño
     - Modelo ganador
     - Parámetros relevantes (hyperparámetros)
     - Semillas usadas (`random_state`)
     - Tamaños de los conjuntos (train/test)
     - Información del dataset (features, etiquetas)

> **Checkpoint Git:**  
> Commit: `feat: script de entrenamiento y comparación de modelos IRIS`.


---

## 3. Etapa Backend: API en Django

### 3.1. Crear proyecto Django

1. Instalar Django y, si quieres, Django REST Framework:
   ```bash
   pip install django djangorestframework
   ```
2. Crear proyecto dentro de carpeta "backend":
   ```bash
   django-admin startproject iris_api
   ```
3. Entrar a `iris_api` y crear una app, por ejemplo `ml_api`:
   ```bash
   python manage.py startapp ml_api
   ```
4. Registrar `ml_api` (y DRF si lo usas) en `INSTALLED_APPS`.

> **Checkpoint Git:**  
> Commit: `feat: proyecto django iris_api con app ml_api`.

### 3.2. Integrar ML al backend

Dentro de `ml_api`:

1. Crear un módulo, por ejemplo `services.py`, que:
   - integrar los pkl que se encuentran en la carpeta models/ que ya estan creado al backend de django. Al iniciar Django, el sistema deberia cargar los pkl.
   - Cargue los modelos al iniciar el servidor (o bajo demanda).
2. Usar Copilot con prompts del tipo:
   - *“Create a Django view that receives a POST with model_name and iris features and returns a JSON prediction using a loaded sklearn model”*.

### 3.3. Endpoints a implementar

1. **Endpoint de predicción con un modelo específico**  
   - Ruta sugerida: `POST /api/predict/`
   - Body JSON:
     ```json
     {
       "model_name": "rf",
       "features": [5.1, 3.5, 1.4, 0.2]
     }
     ```
   - Respuesta:
     ```json
     {
       "model_name": "rf",
       "prediction": "setosa",
       "prediction_proba": [0.9, 0.1, 0.0]
     }
     ```

2. **Endpoint de comparación de todos los modelos**  
   - Ruta sugerida: `GET /api/compare/` o `POST /api/compare/`
   - Utiliza `results.json` para:
     - Listar modelos
     - Mostrar métricas
     - Indicar el ganador
   - Respuesta ejemplo:
     ```json
     {
       "models": [
         {"name": "rf", "accuracy": 0.97},
         {"name": "svc", "accuracy": 0.95}
       ],
       "best_model": "rf"
     }
     ```

3. **Endpoint de XAI (SHAP)**  
   - Ruta sugerida: `POST /api/xai/`
   - Body JSON (misma estructura de features que en `/predict/`).
   - Lógica:
     1. Cargar el modelo seleccionado.
     2. Calcular valores SHAP para esa instancia.
     3. Generar una gráfica SHAP (por ejemplo `force_plot` o `bar_plot`) y guardarla como imagen.
     4. Devolver:
        - La imagen en base64 o URL.
        - Un texto explicativo de alto nivel (puede apoyarse en Copilot).

   - Respuesta ejemplo:
     ```json
     {
       "model_name": "rf",
       "shap_image_base64": "<string>",
       "explanation": "For this prediction, petal_length and petal_width contributed the most..."
     }
     ```

### 3.4. Pruebas con Postman

1. Configurar **Postman** con una colección de requests:
   - `POST /api/predict/`
   - `GET/POST /api/compare/`
   - `POST /api/xai/`
2. Probar:
   - Casos válidos.
   - Errores típicos (falta de parámetros, modelo inexistente, etc.).

> **Checkpoint Git:**  
> Commit: `feat: api django con endpoints de predicción, comparación y XAI`  
> (idealmente con tests básicos o al menos requests de Postman documentadas).

---

## 4. Etapa Frontend: React + Vite + TailwindCSS

### 4.1. Crear proyecto Vite + React

1. Asegurarse de tener **Node.js** instalado.
2. Crear el proyecto (JavaScript):
   ```bash
   npm create vite@latest iris-frontend -- --template react
   cd iris-frontend
   npm install
   ```
3. Ejecutar para probar:
   ```bash
   npm run dev
   ```
   Verificar que la página base de Vite funciona.

> **Checkpoint Git (en subcarpeta o repo separado):**  
> Commit: `feat: proyecto vite react base`.

### 4.2. Instalar y configurar TailwindCSS

1. Seguir los pasos estándar de instalación de TailwindCSS para Vite.
2. Configurar `tailwind.config.js` y agregar las directivas en `index.css`.
3. Crear algunos componentes UI básicos (botones, formularios) para probar estilos.

> **Checkpoint Git:**  
> Commit: `chore: configuración de tailwindcss en vite react`.

### 4.3. Consumir la API desde React

Componentes sugeridos:

1. **Formulario de predicción (`PredictionForm`)**
   - Inputs numéricos para las 4 features de IRIS.
   - Selector de modelo (`rf`, `svc`).
   - Botón “Predecir”.
   - Llamar a `POST /api/predict/` con `fetch` o `axios`.
   - Mostrar la clase predicha y, opcionalmente, probabilidades.

2. **Vista de comparación (`ComparisonView`)**
   - Botón “Actualizar comparación”.
   - Llamar a `/api/compare/`.
   - Mostrar tabla con:
     - Modelo
     - Accuracy (y otras métricas si se incluyen)
     - Indicar visualmente el modelo ganador.

3. **Vista de XAI (`XaiView`)**
   - Reutiliza el formulario de features.
   - Llama a `/api/xai/`.
   - Muestra la imagen SHAP (decodificando base64 en un `<img>`).
   - Muestra el texto explicativo debajo (“por qué el modelo tomó la decisión”).

> **Checkpoint Git:**  
> Commit: `feat: integración frontend con api de predicción, comparación y XAI`.

---

## 5. Integración y ejecución de punta a punta

1. **Backend (Django)**  
   ```bash
   cd iris_api
   python manage.py runserver
   ```
   Servir en `http://127.0.0.1:8000/`.

2. **Frontend (Vite/React)**  
   ```bash
   cd iris-frontend
   npm run dev
   ```
   Servir en `http://localhost:5173/` (u otro puerto que indique Vite).

3. En el frontend, configurar la URL base de la API (ej. `http://127.0.0.1:8000/api/`).

4. Probar el flujo completo:
   - Ingresar features → obtener predicción.
   - Ver comparación de modelos.
   - Ver explicación XAI.

---

## 6. Buenas prácticas de Git/GitHub (evitar problemas de rollback)

Dado el problema anterior (faltó granularidad en los commits), se recomienda:

1. **Commits pequeños y frecuentes**, al finalizar cada sub-etapa:
   - Core ML
   - API básica
   - Endpoint XAI
   - Integración con frontend, etc.
2. **Etiquetas (tags)** en los hitos clave:
   - `v0.1-ml-core`
   - `v0.2-backend-api`
   - `v0.3-frontend-integration`
3. **Ramas experimentales** para cambios grandes:
   - Ej. `feature/new-xai-plot`, `feature/add-new-model`.
4. **Push regular a GitHub** para tener backup y poder volver a versiones estables.
