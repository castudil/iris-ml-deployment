from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json

from .services import model_service


@require_http_methods(["GET"])
def health_check(request):
    """
    Endpoint simple para verificar que la API está funcionando
    y que los modelos están cargados
    """
    available_models = model_service.get_available_models()
    metadata = model_service.get_metadata()
    
    return JsonResponse({
        "status": "ok",
        "message": "ML API is running",
        "models_loaded": len(available_models),
        "available_models": available_models,
        "metadata_loaded": metadata is not None
    })


@require_http_methods(["GET"])
def list_models(request):
    """
    Endpoint para listar todos los modelos disponibles con sus métricas
    """
    metadata = model_service.get_metadata()
    available_models = model_service.get_available_models()
    
    if not metadata:
        return JsonResponse({
            "error": "No metadata available"
        }, status=500)
    
    return JsonResponse({
        "available_models": available_models,
        "models": metadata.get("models", []),
        "dataset": metadata.get("dataset", {}),
        "split": metadata.get("split", {})
    })


@csrf_exempt
@require_http_methods(["POST"])
def predict(request):
    """
    Endpoint para hacer predicciones con un modelo específico
    
    Body JSON esperado:
    {
        "model_name": "rf" o "svc",
        "features": [5.1, 3.5, 1.4, 0.2]
    }
    """
    try:
        data = json.loads(request.body)
        model_name = data.get("model_name")
        features = data.get("features")
        
        if not model_name or features is None:
            return JsonResponse({
                "error": "Missing required fields: model_name and features"
            }, status=400)
        
        # Intentar obtener el modelo
        model = model_service.get_model(model_name)
        
        if model is None:
            return JsonResponse({
                "error": f"Model '{model_name}' not found",
                "available_models": model_service.get_available_models()
            }, status=404)
        
        # Validar features
        if not isinstance(features, list) or len(features) != 4:
            return JsonResponse({
                "error": "features must be a list of 4 numeric values"
            }, status=400)
        
        # Hacer la predicción
        import numpy as np
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        
        # Obtener probabilidades si el modelo lo soporta
        prediction_proba = None
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(features_array)[0].tolist()
        
        # Obtener el nombre de la clase desde metadata
        metadata = model_service.get_metadata()
        target_names = metadata.get("dataset", {}).get("target_names", [])
        prediction_class = target_names[int(prediction)] if target_names else str(prediction)
        
        return JsonResponse({
            "model_name": model_name,
            "prediction": prediction_class,
            "prediction_index": int(prediction),
            "prediction_proba": prediction_proba,
            "features": features
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            "error": "Invalid JSON"
        }, status=400)
    except Exception as e:
        return JsonResponse({
            "error": str(e)
        }, status=500)

