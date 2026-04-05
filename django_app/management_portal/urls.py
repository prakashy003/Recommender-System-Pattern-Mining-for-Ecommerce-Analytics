from django.urls import path

from . import views

app_name = "management_portal"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("users/", views.users_view, name="users"),
    path("recommendations/", views.recommendations_view, name="recommendations"),
    path("rules/", views.rules_view, name="rules"),
    path("results/", views.results_view, name="results"),
]
