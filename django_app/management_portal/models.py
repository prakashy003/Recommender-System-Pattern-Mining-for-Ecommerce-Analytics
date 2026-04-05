from django.db import models


class InteractionLog(models.Model):
	user_id = models.CharField(max_length=100)
	product_id = models.CharField(max_length=100)
	rating = models.FloatField()
	created_at = models.DateTimeField(auto_now_add=True)

	class Meta:
		ordering = ["-created_at"]

	def __str__(self) -> str:
		return f"{self.user_id} -> {self.product_id} ({self.rating})"
