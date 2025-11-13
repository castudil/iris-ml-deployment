"""
URLs para la app ml_api
"""
from django.urls import path
from . import views

urlpatterns = [
    path('health/', views.health_check, name='health_check'),
    path('models/', views.list_models, name='list_models'),
    path('predict/', views.predict, name='predict'),
]
