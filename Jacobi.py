import numpy as np
from numpy.linalg import norm
from Refactoring.colors import bcolors
from Refactoring.matrix_utility import is_diagonally_dominant


def jacobi_iterative(A, b, X0=None, TOL=1e-16, N=200):

    # Initialize if X0 is not provided
    if X0 is None:
        X0 = np.zeros_like(b, dtype=np.double)

    if is_diagonally_dominant(A):
        print('Matrix is diagonally dominant - performing Jacobi algorithm\n')

    print_iteration_header(A)

    for k in range(1, N + 1):
        x = perform_jacobi_iteration(A, b, X0)

        print_iteration(k, x)

        if has_converged(x, X0, TOL):
            return tuple(x)

        X0 = x.copy()

    print("Maximum number of iterations exceeded")
    return tuple(x)


def perform_jacobi_iteration(A, b, X0):
    n = len(A)
    x = np.zeros(n, dtype=np.double)

    for i in range(n):
        sigma = sum(A[i][j] * X0[j] for j in range(n) if j != i)
        x[i] = (b[i] - sigma) / A[i][i]

    return x


def has_converged(x, X0, TOL):
    return norm(x - X0, np.inf) < TOL


def print_iteration_header(A):
    iteration_vars = ["x{}".format(i) for i in range(1, len(A) + 1)]
    print("Iteration" + "\t\t".join([" {:>12}".format(var) for var in iteration_vars]))
    print("-----------------------------------------------------------------------------------------------")


def print_iteration(k, x):
    print("{:<15} ".format(k) + "\t\t".join(["{:<15} ".format(val) for val in x]))


if __name__ == "__main__":
    A = np.array([[3, -1, 1], [0, 1, -1], [1, 1, -2]])
    b = np.array([4, -1, -3])

    X0 = np.zeros_like(b, dtype=np.double)
    solution = jacobi_iterative(A, b, X0)

    print(bcolors.OKBLUE, "\nApproximate solution:", solution)
