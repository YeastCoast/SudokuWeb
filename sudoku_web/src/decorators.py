import time
from functools import wraps

def time_wrapper(func):
    wraps(func)
    def wrapper(*args, **kwargs):
        a = time.time_ns()
        result = func(*args, **kwargs)
        b = time.time_ns()
        print(f'Function took {b-a} nsecs.')
        return result
    return wrapper

class CountCalls:
    def __init__(self, func):
        self.func = func
        self.num_calls = 0

    def __call__(self, *args, **kwargs):
        self.num_calls += 1
        #print(f'this is executed {self.num_calls} times')
        return self.func(*args, **kwargs)