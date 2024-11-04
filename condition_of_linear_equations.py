import numpy as np
from Refactoring.L2.inverse_matrix import matrix_inverse
from Refactoring.colors import bcolors
from Refactoring.matrix_utility import print_matrix


def calculate_norm(matrix):
    return max(sum(abs(matrix[row][col]) for col in range(len(matrix))) for row in range(len(matrix)))


def print_matrix_with_label(matrix, label):
    print(bcolors.OKBLUE, f"{label}:", bcolors.ENDC)
    print_matrix(matrix)


def condition_number(A):
    norm_A = calculate_norm(A)  # Step 1: Calculate the infinity norm of A
    A_inv = matrix_inverse(A)  # Step 2: Calculate the inverse of A
    norm_A_inv = calculate_norm(A_inv)  # Step 3: Calculate the infinity norm of the inverse of A

    # Step 4: Compute the condition number
    cond_number = norm_A * norm_A_inv

    # Print matrices and intermediate values
    print_matrix_with_label(A, "A")
    print_matrix_with_label(A_inv, "inverse of A")

    print(bcolors.OKBLUE, "Max Norm of A:", bcolors.ENDC, norm_A)
    print(bcolors.OKBLUE, "Max Norm of the inverse of A:", bcolors.ENDC, norm_A_inv)

    return cond_number


if __name__ == '__main__':
    A = np.array([[2, 1.7, -2.5],
                  [1.24, -2, -0.5],
                  [3, 0.2, 1]])

    cond = condition_number(A)
    print(bcolors.OKGREEN, "\nCondition number:", cond, bcolors.ENDC)
