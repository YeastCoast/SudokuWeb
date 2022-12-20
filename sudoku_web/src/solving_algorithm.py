import math
import numpy as np
from .decorators import CountCalls, time_wrapper


def block_coords(coord):
    block_row = math.floor(coord[0] / 3)
    block_col = math.floor(coord[1] / 3)
    for i in range(block_row * 3, (block_row + 1) * 3):
        for j in range(block_col * 3, (block_col + 1) * 3):
            yield i, j


def col_coords(coord):
    for i in range(0, 9):
        yield i, coord[1]


def row_coords(coord):
    for i in range(0, 9):
        yield coord[0], i


def effect_range(coord, pending_coords):
    affected_coords = []
    checkers = [block_coords(coord), row_coords(coord), col_coords(coord)]
    for checker in checkers:
        for i in checker:
            if i not in affected_coords and i in pending_coords and i != coord:
                affected_coords.append(i)
    return affected_coords


def get_possible_values(coord, grid):
    all_possible_values = np.asarray(range(1, 10))
    checkers = (block_coords(coord), row_coords(coord), col_coords(coord))
    blocked_vals = []
    for checker in checkers:
        for j in checker:
            val = grid[j[0], j[1]]
            if val not in blocked_vals:
                vals = grid[j[0], j[1]]
                blocked_vals.append(val)
    pos_vals = set(all_possible_values).difference(set(blocked_vals))
    return pos_vals


@time_wrapper
def solve_grid(grid, coords):
    used = []
    poss_vals = {tuple(i): get_possible_values(i, grid) for i in coords}
    poss_vals_amount = {i: len(poss_vals[i]) for i in poss_vals}
    start_coord = min(poss_vals_amount, key=poss_vals_amount.get)

    @CountCalls
    def helper(coord, poss_vals):
        if helper.num_calls % 1000 == 0:
            pass
            print(grid)
        if len(used) == len(coords):
            #print('finished')
            return 0
        used.append(coord)
        for val in poss_vals:
            grid[coord[0], coord[1]] = val
            poss_vals_new = {tuple(i): get_possible_values(i, grid) for i in coords if tuple(i) not in used}
            poss_vals_amount_new = {i: len(poss_vals_new[i]) for i in poss_vals_new}
            if any([i == 0 for i in poss_vals_amount_new.values()]):
                grid[coord[0], coord[1]] = 0
                continue
            if len(used) == len(coords):
                #print('finished')
                return 0
            coord_new = min(poss_vals_amount_new, key=poss_vals_amount_new.get)
            return_val = helper(coord_new, poss_vals_new[coord_new])
            if return_val == 0:
                return 0
        grid[coord[0], coord[1]] = 0
        used.pop()
        return 1


    helper(start_coord, poss_vals[start_coord])
    print(f'this is executed {helper.num_calls} times')
    return grid


def check_input(grid):
    def check_unqiue(entry):
        curr_entry = entry[entry != 0]
        if curr_entry is []:
            return True
        else:
            a, c = np.unique(curr_entry, return_counts=True)
            if any(c > 1):
                return False
            else:
                return True

    if len(grid[grid == 0]) == 0:
        return False

    for row in grid:
        bool_val = check_unqiue(row)
        if not bool_val:
            return False

    for col in grid.T:
        bool_val = check_unqiue(col)
        if not bool_val:
            return False

    for row_id in range(0, len(grid), 3):
        for col_id in range(0, len(grid[row_id]), 3):
            curr_entry = grid[row_id:row_id+3, col_id:col_id+3]
            curr_entry = np.reshape(curr_entry, (1, 9))[0]
            bool_val = check_unqiue(curr_entry)
            if not bool_val:
                return False

    return True

def solver(grid):
    coords = np.argwhere(grid == 0)
    if check_input(grid):
        solved_grid = solve_grid(grid, coords)
        #print(solved_grid)
        return solved_grid, coords
    else:
        return grid, coords


if __name__ == '__main__':
    test_grid = [[0, 0, 0, 6, 0, 2, 8, 0, 4],
             [0, 0, 0, 0, 3, 0, 0, 0, 7],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [4, 0, 6, 0, 5, 0, 3, 0, 0],
             [2, 0, 8, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 9, 1, 0],
             [1, 0, 0, 0, 0, 0, 2, 0, 0],
             [0, 7, 0, 9, 0, 0, 0, 5, 0],
             [0, 0, 2, 4, 0, 0, 0, 0, 8]]

    solving_grid = np.asarray(test_grid)

    solver(solving_grid)

    #print(solving_grid)
