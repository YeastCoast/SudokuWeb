from django.urls import path
from . import views


app_name = 'sudoku_web'
urlpatterns = [
    path('', views.IndexView.as_view(), name='index'),
    path('Image/', views.ImageView.as_view(), name='image'),
    path('Manual/', views.ManualView.as_view(), name='manual')
]