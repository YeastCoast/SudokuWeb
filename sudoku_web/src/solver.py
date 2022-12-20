from sudoku_web.src.image_parser import grid_recognition, digit_recognition
from sudoku_web.src.solving_algorithm import solver
from sudoku_web.src.visualizer import draw_digits, unwarp_image, find_contours
from io import BytesIO
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import base64


def solve_sudoku(img):
    model_path = 'sudoku_web/src/model/model.hdf5'
    #model_path = 'model/model.hdf5'
    if isinstance(img, BytesIO):
        image = cv2.imdecode(np.frombuffer(img.read(), np.uint8), 1)
    else:
        image = cv2.imread(img)
    img_result = image.copy()
    transformed_grid, corners = grid_recognition(image)
    #plt.imshow(transformed_grid)
    #plt.show()
    sudoku_array = digit_recognition(transformed_grid, model_path)
    #print(sudoku_array)
    solved_array, non_zero_coords = solver(sudoku_array)
    solved_warped = draw_digits(solved_array, non_zero_coords, transformed_grid)
    solved_unwarped = unwarp_image(solved_warped, img_result, corners)
    output = Image.fromarray(solved_unwarped, 'RGB')
    data = BytesIO()
    output.save(data, "JPEG")  # pick your format
    data64 = base64.b64encode(data.getvalue())
    return u'data:img/jpeg;base64,' + data64.decode('utf-8')


def grid_to_array(django_grid: dict):
    django_default = {i: int(django_grid[i]) if django_grid[i].isdigit() else 0 for i in django_grid}
    lst = [[django_default[f"{i}_{j}"] for j in range(9)] for i in range(9)]
    sudoku_array = np.array(lst)
    return sudoku_array


def array_to_grid(sudoku_array: np.ndarray):
    return {f"{i}_{j}": str(sudoku_array[i][j]) for i in range(len(sudoku_array)) for j in range(len(sudoku_array[i]))}


def solve_grid(django_grid: dict):
    sudoku_array = grid_to_array(django_grid)
    solved_array, non_zero_coords = solver(sudoku_array)
    solved_grid = array_to_grid(solved_array)

    return solved_grid
