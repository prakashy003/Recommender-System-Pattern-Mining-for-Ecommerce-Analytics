from django import forms


class AddInteractionForm(forms.Form):
    user_id = forms.CharField(max_length=100, label="User ID")
    product_id = forms.CharField(max_length=100, label="Product ID")
    rating = forms.FloatField(min_value=1.0, max_value=5.0, label="Rating")
