import numpy as np
from Refactoring.colors import bcolors
from Refactoring.matrix_utility import swap_rows_elementary_matrix, row_addition_elementary_matrix


def lu_decomposition(A):
    N = len(A)
    L = np.eye(N)  # Initialize L as the identity matrix

    for i in range(N):
        pivot_row = find_pivot_row(A, i, N)
        A = handle_pivot_row(A, L, i, pivot_row, N)

        for j in range(i + 1, N):
            A, L = eliminate_below_pivot(A, L, i, j, N)

    U = A
    return L, U


def find_pivot_row(A, i, N):
    pivot_row = i
    v_max = A[i][i]
    for j in range(i + 1, N):
        if abs(A[j][i]) > abs(v_max):
            v_max = A[j][i]
            pivot_row = j
    return pivot_row


def handle_pivot_row(A, L, i, pivot_row, N):
    if A[i][pivot_row] == 0:
        raise ValueError("can't perform LU Decomposition")

    if pivot_row != i:
        e_matrix = swap_rows_elementary_matrix(N, i, pivot_row)
        print_elementary_matrix(e_matrix, f"swap between row {i} and row {pivot_row}")
        A = np.dot(e_matrix, A)
        print_matrix(A)

    return A


def eliminate_below_pivot(A, L, i, j, N):
    m = -A[j][i] / A[i][i]
    e_matrix = row_addition_elementary_matrix(N, j, i, m)
    e_inverse = np.linalg.inv(e_matrix)

    L = np.dot(L, e_inverse)
    A = np.dot(e_matrix, A)

    print_elementary_matrix(e_matrix, f"zero the element in row {j} below pivot in column {i}")
    print_matrix(A)

    return A, L


def print_elementary_matrix(matrix, description):
    print(f"Elementary matrix for {description}:\n{matrix}\n")


def print_matrix(matrix):
    print(f"The matrix after elementary operation:\n{matrix}")
    print(bcolors.OKGREEN, "---------------------------------------------------------------------------", bcolors.ENDC)


def backward_substitution(U):
    N = len(U)
    x = np.zeros(N)

    for i in range(N - 1, -1, -1):
        x[i] = U[i][N]
        for j in range(i + 1, N):
            x[i] -= U[i][j] * x[j]
        x[i] /= U[i][i]

    return x


def lu_solve(A_b):
    L, U = lu_decomposition(A_b)
    print(bcolors.OKBLUE, "Lower triangular matrix L:", bcolors.ENDC)
    print_matrix(L)

    print(bcolors.OKBLUE, "Upper triangular matrix U:", bcolors.ENDC)
    print_matrix(U)

    result = backward_substitution(U)
    print_solution(result)


def print_solution(result):
    print(bcolors.OKBLUE, "\nSolution for the system:", bcolors.ENDC)
    for x in result:
        print("{:.6f}".format(x))


if __name__ == '__main__':
    A_b = [[1, -1, 2, -1, -8],
           [2, -2, 3, -3, -20],
           [1, 1, 1, 0, -2],
           [1, -1, 4, 3, 4]]

    lu_solve(A_b)
