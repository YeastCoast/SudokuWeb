from django.db import models


class SudokuFile(models.Model):
    upload = models.ImageField(upload_to='images/%Y/%m/%d/')
