import numbers

import numpy as np


def is_number(value):
    return isinstance(value, numbers.Number)


def integral(f, a, b, n=2000):
    """Calculates midpoint Riemann sum of f(x) over [a, b]."""
    dx = (b-a)/n
    x = np.arange(a + dx / 2, b + dx / 2, dx)
    y = f(x)
    return np.sum(y*dx), y, x, dx


def derivative(delta_x, y):
    """Calculates numerical derivative of y using central difference."""
    ones = np.ones(len(y) - 1)
    M1 = np.diag(ones, k=1)
    M2 = np.diag(-ones, k=-1)
    M = M1 + M2
    dydx = M @ y / (2 * delta_x)
    return dydx[1:-1]
    

def main():
    f = lambda x: x ** 2
    Y, y, _, _ = integral(f, 0, 1)
    print(f"The integral of f(x) from 0 to 1 is approximately {Y:.5f}.")
    

if __name__ == "__main__":
    main()
    