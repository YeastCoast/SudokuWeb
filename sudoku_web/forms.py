from django import forms
from .models import SudokuFile


class UploadSudokuForm(forms.ModelForm):
    class Meta:
        model = SudokuFile
        fields = {'upload'}
        labels = {'upload': ''}


class SudokuGridForm(forms.Form):
    def __init__(self, *args, **kwargs):
        dynamic_fields = kwargs.pop('dynamic_fields')
        super(SudokuGridForm, self).__init__(*args, **kwargs)
        for key in dynamic_fields:
            self.fields[f"{key}"] = forms.CharField(max_length=1, required=False)
            self.fields[key].widget.attrs['class'] = 'sudoku-cell'
            self.fields[key].label = ''

