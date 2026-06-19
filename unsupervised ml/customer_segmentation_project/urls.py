"""
URL configuration for customer_segmentation_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index_view, name='index'),
    path('api/data', views.get_data_api, name='get_data_api'),
    path('api/elbow', views.get_elbow_api, name='get_elbow_api'),
    path('api/train', views.train_api, name='train_api'),
    path('api/predict', views.predict_api, name='predict_api'),
    path('api/upload', views.upload_api, name='upload_api'),
    path('api/reset', views.reset_api, name='reset_api'),
]
