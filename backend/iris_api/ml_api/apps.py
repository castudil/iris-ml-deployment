from django.apps import AppConfig


class MlApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ml_api'
    
    def ready(self):
        """
        Se ejecuta cuando Django se inicia.
        Aquí cargamos los modelos PKL.
        """
        print("\n" + "="*50)
        print("🚀 Inicializando ML API...")
        print("="*50)
        
        # Importar y cargar los modelos
        from .services import model_service
        
        # El modelo ya se carga en el __init__ del singleton,
        # pero podemos mostrar información adicional aquí
        available_models = model_service.get_available_models()
        
        if available_models:
            print(f"✓ Modelos disponibles: {', '.join(available_models)}")
        else:
            print("⚠️  No se cargaron modelos")
        
        print("="*50 + "\n")
