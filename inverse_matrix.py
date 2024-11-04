from Refactoring.colors import bcolors
from Refactoring.matrix_utility import row_addition_elementary_matrix, scalar_multiplication_elementary_matrix
import numpy as np


def validate_square_matrix(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")


def check_singular(matrix, i):
    if matrix[i, i] == 0:
        raise ValueError("Matrix is singular, cannot find its inverse.")


def scale_row_to_one(matrix, identity, i):
    scalar = 1.0 / matrix[i, i]
    elementary_matrix = scalar_multiplication_elementary_matrix(matrix.shape[0], i, scalar)
    print(f"Elementary matrix to make the diagonal element 1:\n {elementary_matrix} \n")

    matrix = np.dot(elementary_matrix, matrix)
    identity = np.dot(elementary_matrix, identity)
    print(f"The matrix after elementary operation:\n {matrix}")
    print(bcolors.OKGREEN, "-" * 118, bcolors.ENDC)

    return matrix, identity


def zero_out_elements(matrix, identity, i, n):
    for j in range(n):
        if i != j:
            scalar = -matrix[j, i]
            elementary_matrix = row_addition_elementary_matrix(n, j, i, scalar)
            print(f"Elementary matrix for R{j + 1} = R{j + 1} + ({scalar}R{i + 1}):\n {elementary_matrix} \n")

            matrix = np.dot(elementary_matrix, matrix)
            identity = np.dot(elementary_matrix, identity)

            print(f"The matrix after elementary operation:\n {matrix}")
            print(bcolors.OKGREEN, "-" * 118, bcolors.ENDC)

    return matrix, identity


def matrix_inverse(matrix):
    print(bcolors.OKBLUE,
          f"=================== Finding the inverse of a non-singular matrix using elementary row operations ===================\n {matrix}\n",
          bcolors.ENDC)

    validate_square_matrix(matrix)

    n = matrix.shape[0]
    identity = np.identity(n)

    # Perform row operations to transform the matrix into the identity matrix
    for i in range(n):
        check_singular(matrix, i)

        if matrix[i, i] != 1:
            matrix, identity = scale_row_to_one(matrix, identity, i)

        matrix, identity = zero_out_elements(matrix, identity, i, n)

    return identity


if __name__ == '__main__':
    A = np.array([[1, 2, 3],
                  [2, 3, 4],
                  [3, 4, 6]])

    try:
        A_inverse = matrix_inverse(A)
        print(bcolors.OKBLUE, "\nInverse of matrix A: \n", A_inverse)
        print("=" * 120, bcolors.ENDC)
    except ValueError as e:
        print(str(e))
