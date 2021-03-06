from django.conf.urls import url
from . import views
from users import models

urlpatterns = [
    url(r'^$', views.listing, name="files-listing"),
    url(r'^list$', views.lists, name="files-lists"),
    url(r'^add$', views.add, name="add"),
    url(r'^delete/(?P<id>\w{0,50})/$', views.delete, name="delete"),

    # url(r'^gcookie$', views.gcookie, name='gcookie')
]
