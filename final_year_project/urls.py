from django.contrib import admin
from django.urls import path, include
from django.conf.urls.i18n import i18n_patterns
from core import views
urlpatterns = [
    path("", include("core.urls")),
    path("admin/", admin.site.urls),
    path('i18n/', include('django.conf.urls.i18n')),
]
urlpatterns += i18n_patterns(
    path('', views.dashboard, name='dashboard'),
)