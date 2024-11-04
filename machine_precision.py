from Refactoring.colors import bcolors


def get_machine_epsilon():
    eps = 1
    while is_epsilon_effective(eps):
        eps /= 2
    return eps * 2


def is_epsilon_effective(eps):
    return (1 + eps) > 1


def calculate_expression():
    return abs(3.0 * (4.0 / 3.0 - 1) - 1)


def display_results(expression, machine_eps):
    print_result("Machine Precision  :", machine_eps, bcolors.OKBLUE)

    print("\nResult of abs(3.0 * (4.0 / 3.0 - 1) - 1) :")
    print_result("before using machine epsilon", expression, bcolors.FAIL)
    print_result("After correcting with machine epsilon", expression - machine_eps, bcolors.OKGREEN)


def print_result(description, value, color):
    print(color, "{}: {}".format(description, value), bcolors.ENDC)


if __name__ == '__main__':
    machine_eps = get_machine_epsilon()
    expression = calculate_expression()

    display_results(expression, machine_eps)
