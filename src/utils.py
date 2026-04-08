import numbers

import numpy as np


def is_number(value):
    return isinstance(value, numbers.Number)


def is_singular(A):
    """Checks if a matrix A is singular by computing its determinant."""
    return np.linalg.det(A) == 0


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


def lu_decomposition(A):
    """Performs LU decomposition of a square matrix A."""
    # Check input type by converting to numpy array
    try:
        A = np.array(A, dtype=float)
    except Exception:
        raise TypeError(
            "Input must be convertible to a numpy array of numbers."
        )
    # Check for empty input
    if A.shape[0] == 0:
        raise ValueError("Input matrix A cannot be empty.")
    
    # Handle last iteration (1x1 matrix)
    if A.shape[0] == 1:
        return 1, A
    
    # Otherwise check if A is square
    if A.shape[0] != A.shape[1]:
        raise ValueError("Input matrix A must be square.")
    
    # Upper matrix first row
    u11 = A[0, 0]
    uV = A[0, 1:]
    
    # Lower matrix first column
    l11 = 1
    lV = A[1:, 0] / u11
    
    # Schur complement
    S = A[1:, 1:] - np.outer(lV, uV)
    
    # LU decomposition of Schur complement
    L22, U22 = lu_decomposition(S)
    
    # Construct L and U
    U = np.zeros_like(A)
    U[0, 0] = u11
    U[0, 1:] = uV
    U[1:, 1:] = U22
    
    L = np.zeros_like(A)
    L[0, 0] = l11
    L[1:, 0] = lV
    L[1:, 1:] = L22
    
    return L, U


def main():
    A = [[1, 2], [3, 4]]
    print(lu_decomposition(A))
    

if __name__ == "__main__":
    main()
    