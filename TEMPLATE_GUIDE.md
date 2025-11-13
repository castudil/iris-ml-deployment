# 🚀 ML Deployment Template

Template para despliegue de proyectos de Machine Learning con Django + React + Docker.

## 📋 Características

- ✅ Backend Django con REST API
- ✅ Frontend React con Tailwind CSS
- ✅ Carga automática de modelos ML (PKL)
- ✅ Docker y Docker Compose configurados
- ✅ CORS configurado
- ✅ Health checks
- ✅ XAI ready (preparado para SHAP)

## 🔄 Cómo usar este template para un nuevo proyecto

### 1. Clonar y renombrar

```bash
# Opción A: Copiar la carpeta
cp -r iris-ml-deployment mi-nuevo-proyecto
cd mi-nuevo-proyecto

# Opción B: Usar como template en GitHub
# (Crear nuevo repo desde template en GitHub)
```

### 2. Actualizar nombres del proyecto

Buscar y reemplazar en todos los archivos:

- `iris-ml-deployment` → `tu-nuevo-proyecto`
- `iris_api` → `tu_api`
- `ml_api` → `tu_app_api`
- `Iris` → `TuDominio`

### 3. Adaptar el modelo de datos

**Backend (`ml_api/services.py`):**
- Mantener la estructura de carga de modelos PKL
- Actualizar metadata esperada en `results.json`
- Adaptar número y nombres de features

**Frontend:**
- Actualizar `PredictionForm.jsx` con tus features
- Modificar `ResultCard.jsx` para mostrar tus resultados
- Personalizar colores y categorías

### 4. Entrenar tus modelos

```bash
# Crear script de entrenamiento (similar a train_models.py)
python train_your_models.py

# Generar:
# - models/*.pkl (tus modelos entrenados)
# - models/results.json (metadata)
```

### 5. Configurar Docker

```bash
# Limpiar datos anteriores
rm -rf models/*.pkl models/results.json

# Copiar tus modelos
cp /ruta/tus/modelos/*.pkl models/

# Build y deploy
docker-compose up -d --build
```

## 📁 Estructura del template

```
project/
├── backend/
│   ├── Dockerfile
│   └── {project}_api/
│       ├── manage.py
│       └── ml_api/
│           ├── services.py    # ← Adaptar carga de modelos
│           └── views.py       # ← Adaptar endpoints
├── frontend/
│   ├── Dockerfile
│   ├── src/
│   │   ├── components/
│   │   │   ├── PredictionForm.jsx  # ← Personalizar inputs
│   │   │   └── ResultCard.jsx      # ← Personalizar outputs
│   │   └── config/
│   │       └── api.js
│   └── nginx.conf
├── models/                    # ← Tus modelos PKL aquí
├── docs/
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🔧 Componentes reutilizables

### ✅ NO modificar (funciona igual):
- Docker setup completo
- Nginx configuración
- CORS setup
- Health checks
- Sistema de carga de modelos PKL
- Arquitectura de servicios

### 🔄 Personalizar por proyecto:
- Número y tipo de features
- Categorías/clases de predicción
- Visualizaciones de resultados
- Colores y branding
- Endpoints adicionales (XAI, comparación)

## 📝 Checklist de adaptación

- [ ] Renombrar proyecto
- [ ] Actualizar requirements.txt con tus dependencias
- [ ] Entrenar y guardar modelos en `models/`
- [ ] Actualizar `results.json` con metadata
- [ ] Modificar `PredictionForm.jsx` (features)
- [ ] Modificar `ResultCard.jsx` (visualización)
- [ ] Actualizar `Header.jsx` (branding)
- [ ] Ajustar colores en Tailwind
- [ ] Probar localmente
- [ ] Probar con Docker
- [ ] Documentar features específicas

## 🎨 Para tu proyecto de Biometría/Sensores

### Features típicas podrían ser:
```javascript
// Ejemplo: Datos biométricos
{
  age: number,
  heart_rate: number,
  blood_pressure_sys: number,
  blood_pressure_dia: number,
  temperature: number,
  // ... más features
}
```

### Clases/outputs podrían ser:
- Clasificación: "normal", "alerta", "crítico"
- Detección anomalías: "normal", "anomalía"
- Múltiples outputs con SHAP para explicabilidad

## 🚀 Próximos pasos sugeridos

1. **Crear branch `template`** con versión genérica
2. **Crear nuevo proyecto** desde template
3. **Adaptar gradualmente** manteniendo estructura
4. **Agregar XAI** (SHAP) como nuevo endpoint
5. **Escalar** según necesidades

## 💡 Ventajas de este approach

✅ No reinventar la rueda
✅ Docker ya configurado
✅ CORS resuelto
✅ Frontend profesional base
✅ Arquitectura probada
✅ Fácil de escalar

## ⚠️ Advertencias

- Mantener estructura de carpetas
- No cambiar nombres de archivos Docker si no es necesario
- Documentar cambios específicos del dominio
- Versionar con Git desde el inicio
