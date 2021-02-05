import numpy as np
from itertools import repeat
import multiprocessing as mp


def funcdict(a, b, di):
    return di['label'][a] + inner(b)


def inner(b):
    return 2 * b


def funcdict1(di):
    return di['label'] + 1


def fun(a, b):
    return a + b


def func_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return funcdict(*a_b)


def main():
    a_args = [1, 2, 3]
    second_arg = 1
    pool = mp.Pool(processes=4)
    results = pool.starmap(fun, zip(a_args, repeat(second_arg)))
    print(results)

    manager = mp.Manager()
    pydict = {'label': np.arange(1000)}
    d = manager.dict(pydict)
    # d['label'] = np.arange(1000)
    jam = np.arange(1000)
    # results = pool.starmap(partial(funcdict, di=d), jam)
    results = pool.map(func_star, zip(jam, jam + 1, repeat(d)))
    print(len(results), '\n', results)


if __name__ == "__main__":
    main()
