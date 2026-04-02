import numpy as np


def integral(f, a, b, n):
    """Calculates midpoint Riemann integral on an interval."""
    dx = (b-a)/n
    x = np.arange(a + dx / 2, b - dx / 2, dx)
    y = f(x)
    return np.sum(y*dx)


def main():
    f = lambda x: x ** 2
    F = integral(f, 0, 1, 1000)
    print(f"The integral of f(x) from 0 to 1 is approximately {F:.5f}.")
    

if __name__ == "__main__":
    main()
    