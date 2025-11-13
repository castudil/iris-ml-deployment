#!/bin/bash

# Script para adaptar el template a un nuevo proyecto
# Uso: ./rename_project.sh nuevo-nombre NuevoNombre

if [ "$#" -ne 2 ]; then
    echo "Uso: $0 <nuevo-nombre-proyecto> <NombreClase>"
    echo "Ejemplo: $0 biometric-detection BiometricDetection"
    exit 1
fi

NEW_NAME=$1
NEW_CLASS=$2

echo "🔄 Renombrando proyecto a: $NEW_NAME"

# Renombrar en archivos
find . -type f -not -path "*/\.*" -not -path "*/node_modules/*" -not -path "*/venv/*" -not -path "*/__pycache__/*" -exec sed -i '' "s/iris-ml-deployment/$NEW_NAME/g" {} +
find . -type f -not -path "*/\.*" -not -path "*/node_modules/*" -not -path "*/venv/*" -not -path "*/__pycache__/*" -exec sed -i '' "s/iris_api/${NEW_NAME//-/_}_api/g" {} +
find . -type f -not -path "*/\.*" -not -path "*/node_modules/*" -not -path "*/venv/*" -not -path "*/__pycache__/*" -exec sed -i '' "s/Iris/$NEW_CLASS/g" {} +

# Renombrar directorios
if [ -d "backend/iris_api" ]; then
    mv backend/iris_api backend/${NEW_NAME//-/_}_api
    echo "✅ Renombrado backend/iris_api → backend/${NEW_NAME//-/_}_api"
fi

# Limpiar modelos de ejemplo
echo "🧹 Limpiando modelos de ejemplo..."
rm -f models/*.pkl models/results.json

echo ""
echo "✅ Proyecto renombrado exitosamente!"
echo ""
echo "📋 Próximos pasos:"
echo "1. Actualizar README.md con descripción del nuevo proyecto"
echo "2. Entrenar tus modelos y guardarlos en models/"
echo "3. Actualizar frontend/src/components/PredictionForm.jsx con tus features"
echo "4. Actualizar frontend/src/components/ResultCard.jsx con tus visualizaciones"
echo "5. Probar: docker-compose up -d --build"
echo ""
