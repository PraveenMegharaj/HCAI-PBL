from django.urls import path

from . import views

app_name = 'project4'

urlpatterns = [
    path('',            views.landing,       name='index'),
    path('landing/',    views.landing,       name='landing'),
    path('consent/',    views.consent,       name='consent'),
    path('study/',      views.study,         name='study'),
    path('complete/',   views.complete,      name='complete'),
]
