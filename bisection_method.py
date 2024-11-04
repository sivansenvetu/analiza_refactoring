import math
import numpy as np
from Refactoring.colors import bcolors


def max_steps(a, b, err):
    s = int(np.floor(-np.log2(err / (b - a)) / np.log2(2) - 1))
    return s


def bisection_method(f, a, b, tol=1e-6):

    validate_bounds(f, a, b)  # Ensure f(a) and f(b) have opposite signs
    steps = max_steps(a, b, tol)  # Calculate the maximum steps possible
    print_iteration_header()

    c, k = 0, 0

    # Iteratively find the root using the bisection method
    while abs(b - a) > tol and k < steps:
        c = calculate_midpoint(a, b)

        if f(c) == 0:
            return c  # Exact root found

        update_bounds(f, c, a, b)  # Update bounds based on the sign of f(c)

        print_iteration(k, a, b, f(a), f(b), c, f(c))
        k += 1

    return c  # Return the current approximation of the root


def validate_bounds(f, a, b):
    if np.sign(f(a)) == np.sign(f(b)):
        raise ValueError("The scalars a and b do not bound a root")


def calculate_midpoint(a, b):
    return a + (b - a) / 2


def update_bounds(f, c, a, b):
    if f(c) * f(a) < 0:  # Sign change indicates root is in [a, c]
        b = c
    else:  # Sign change indicates root is in [c, b]
        a = c
    return a, b


def print_iteration_header():
    print("{:<10} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format("Iteration", "a", "b", "f(a)", "f(b)", "c", "f(c)"))


def print_iteration(k, a, b, fa, fb, c, fc):
    print("{:<10} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f}".format(k, a, b, fa, fb, c, fc))


if __name__ == '__main__':
    f = lambda x: x ** 2 - 4 * math.sin(x)
    roots = bisection_method(f, 1, 3)
    print(bcolors.OKBLUE, f"\nThe equation f(x) has an approximate root at x = {roots}", bcolors.ENDC)
