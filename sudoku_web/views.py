from django.shortcuts import render, redirect
from django.urls import reverse_lazy
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.views import generic

from .forms import UploadSudokuForm, SudokuGridForm
from .models import SudokuFile
from .src.solver import solve_sudoku, solve_grid

import os


class IndexView(generic.View):
    template_name = 'sudoku_web/index.html'

    def get(self, request):
        return render(request, self.template_name)


class ImageView(generic.View):
    template_name = 'sudoku_web/image.html'

    def get(self, request):
        form = UploadSudokuForm()
        return render(request, self.template_name, {'form': form})

    def post(self, request):
        form = UploadSudokuForm(request.POST, request.FILES)
        if form.is_valid():
            new_image = SudokuFile(upload=request.FILES['upload'])
            new_image.save()
            solved_img = solve_sudoku(new_image.upload.path)
            new_image.delete()
            os.remove(new_image.upload.path)
            form = UploadSudokuForm()
            #return HttpResponseRedirect('SudokuSolver', kwargs={'form': form, 'sudoku': solved_img})
            return render(request, self.template_name, {'form': form, 'sudoku': solved_img})
        else:
            form = UploadSudokuForm()
            return render(request, self.template_name, {'form': form})


class ManualView(generic.View):
    template_name = "sudoku_web/manual.html"
    sudoku_dict = {f"{i}_{j}": 0 for i in range(9) for j in range(9)}

    def get(self, request):
        form = SudokuGridForm(dynamic_fields=self.sudoku_dict)
        return render(request, self.template_name, {'form': form})

    def post(self, request):
        form = SudokuGridForm(request.POST, dynamic_fields=self.sudoku_dict)
        if form.is_valid():

            solved_dict = solve_grid(form.cleaned_data)
            form = SudokuGridForm(solved_dict, dynamic_fields=self.sudoku_dict)

            return render(request, self.template_name, {'form': form})
        else:
            form = SudokuGridForm(dynamic_fields=self.sudoku_dict)
            return render(request, self.template_name, {'form': form})


class redirect_view(generic.View):
    def get(self, request):
        return redirect('/SudokuSolver/')
