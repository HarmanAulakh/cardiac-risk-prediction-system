from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^details/(?P<fileId>\w{0,50})/$', views.details, name="details"),
    url(r'^prediction/(?P<fileId>\w{0,50})/$', views.prediction, name="prediction"),
]
