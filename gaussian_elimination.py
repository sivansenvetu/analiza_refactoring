import numpy as np
from Refactoring.colors import bcolors


def gaussian_elimination(mat):
    N = len(mat)
    singular_flag = forward_substitution(mat)

    if singular_flag != -1:
        return handle_singular_matrix(mat, singular_flag, N)

    return backward_substitution(mat)


def handle_singular_matrix(mat, singular_flag, N):
    if mat[singular_flag][N]:
        return "Singular Matrix (Inconsistent System)"
    else:
        return "Singular Matrix (May have infinitely many solutions)"


def swap_row(mat, i, j):
    mat[i], mat[j] = mat[j], mat[i]


def forward_substitution(mat):
    N = len(mat)
    for k in range(N):
        pivot_row = find_pivot_row(mat, k, N)

        if not mat[pivot_row][k]:
            return k  # Matrix is singular

        if pivot_row != k:
            swap_row(mat, k, pivot_row)

        eliminate_below_pivot(mat, k, N)

    return -1


def find_pivot_row(mat, k, N):
    pivot_row = k
    v_max = mat[k][k]
    for i in range(k + 1, N):
        if abs(mat[i][k]) > abs(v_max):
            v_max = mat[i][k]
            pivot_row = i
    return pivot_row


def eliminate_below_pivot(mat, k, N):
    for i in range(k + 1, N):
        m = mat[i][k] / mat[k][k]
        for j in range(k + 1, N + 1):
            mat[i][j] -= mat[k][j] * m
        mat[i][k] = 0  # Fill lower triangular with zeros


def backward_substitution(mat):
    N = len(mat)
    x = np.zeros(N)

    for i in range(N - 1, -1, -1):
        x[i] = mat[i][N]
        for j in range(i + 1, N):
            x[i] -= mat[i][j] * x[j]
        x[i] /= mat[i][i]

    return x


if __name__ == '__main__':
    A_b = [
        [1, -1, 2, -1, -8],
        [2, -2, 3, -3, -20],
        [1, 1, 1, 0, -2],
        [1, -1, 4, 3, 4]
    ]

    result = gaussian_elimination(A_b)
    if isinstance(result, str):
        print(result)
    else:
        print(bcolors.OKBLUE, "\nSolution for the system:")
        for x in result:
            print("{:.6f}".format(x))
