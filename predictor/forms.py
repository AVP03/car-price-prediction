from django import forms

class PredictionForm(forms.Form):
    engine_power = forms.FloatField(label = 'Engine Power (HP)')