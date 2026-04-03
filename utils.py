import numpy as np


def integral(f, a, b, n=2000):
    """Calculates midpoint Riemann sum of f(x) over [a, b]."""
    dx = (b-a)/n
    x = np.arange(a + dx / 2, b + dx / 2, dx)
    y = f(x)
    return np.sum(y*dx), y, x, dx


def main():
    f = lambda x: x ** 2
    Y, y, _, _ = integral(f, 0, 1)
    print(f"The integral of f(x) from 0 to 1 is approximately {Y:.5f}.")
    

if __name__ == "__main__":
    main()
    