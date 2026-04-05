from django.contrib import admin

from .models import InteractionLog


@admin.register(InteractionLog)
class InteractionLogAdmin(admin.ModelAdmin):
	list_display = ("user_id", "product_id", "rating", "created_at")
	search_fields = ("user_id", "product_id")
	list_filter = ("created_at",)

# Register your models here.
