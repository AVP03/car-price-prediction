from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_price, name='predict_price'),
    path('visualize/', views.visualize_data, name='visualize_data'),
]