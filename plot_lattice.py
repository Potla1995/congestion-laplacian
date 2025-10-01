import numpy
from matplotlib import pyplot as plt


def func_a(dim_m: int, dim_n: int):
    return 4 * min(numpy.sin(numpy.pi / (2 * dim_m)) ** 2, numpy.sin(numpy.pi / (2 * dim_n)) ** 2)


def func_b(dim_m: int, dim_n: int):
    return 4 * min(numpy.sin((dim_m - 1) * numpy.pi / (2 * dim_m)) ** 2, numpy.sin((dim_n - 1) * numpy.pi / (2 * dim_n)) ** 2)


def func_c(dim_m: int, dim_n: int):
    return 4 * min(numpy.sin(numpy.pi / dim_m) ** 2, numpy.sin(numpy.pi / dim_n) ** 2)


def func_d(dim_m: int, dim_n: int):
    largest_cycle_m = 4 if dim_m % 2 == 0 else 4 * numpy.sin((dim_m - 1) * numpy.pi / (2 * dim_m)) ** 2
    largest_cycle_n = 4 if dim_n % 2 == 0 else 4 * numpy.sin((dim_n - 1) * numpy.pi / (2 * dim_n)) ** 2
    return min(largest_cycle_m, largest_cycle_n)


def lower_bd_main_thm(dim_m: int, dim_n: int, periodicity: bool):
    return 2 * dim_m * dim_n * func_c(dim_m, dim_n) / 9 if periodicity else 2 * dim_m * dim_n * func_a(dim_m, dim_n) / 9


def upper_bd_main_thm(dim_m: int, dim_n: int, periodicity: bool):
    lambda_2 = func_c(dim_m, dim_n) if periodicity else func_a(dim_m, dim_n)
    lambda_n = func_d(dim_m, dim_n) if periodicity else func_b(dim_m, dim_n)
    term1 = 0.5 * numpy.sqrt((8 - lambda_2) * lambda_2)
    term2 = (3 / 16 + 1 / (4 * dim_m * dim_n)) * lambda_n
    if term1 >= term2:
        print(f'Lattice[{dim_m}x{dim_n},{periodicity}]: term1 is larger')
    return max(term1, term2) *  dim_m * dim_n


def upper_bd_treewidth(dim_m: int, dim_n: int, periodicity: bool):
    c = 8 if periodicity else 4
    return c * min(dim_m, dim_n) + 3


# MATPLOTLIB PARAMS
params = {"ytick.color": "black",
          "xtick.color": "black",
          "axes.labelcolor": "black",
          "axes.edgecolor": "black",
          "axes.labelpad": 10,
          "axes.labelsize": 16,
          "text.usetex": False,
          "font.family": "serif",
          # "font.serif": ["Computer Modern Roman"],
          'xtick.major.pad': 8,
          'ytick.major.pad': 8}
plt.rcParams.update(params)


# EXPERIMENT 1: Keep m fixed and increase n
m = 5
n_array = list(range(m, 21))
for p in [True, False]:
    plt.figure(1, figsize=(5,3))
    plt.ylim(-10,140)
    lower_main = [lower_bd_main_thm(m, n, p) for n in n_array]
    upper_main = [upper_bd_main_thm(m, n, p) for n in n_array]
    upper_treewidth = [upper_bd_treewidth(m, n, p) for n in n_array]
    lower_treewidth = [(i-4)/4 for i in upper_treewidth]
    print(f'Lower main:{lower_main}\nUpper main:{upper_main}\nUpper tw:{upper_treewidth}')
    plt.plot(n_array, lower_main, 'o-', label='Theorem 1.1 lower bound (Eq. (25))')
    plt.plot(n_array, upper_main, 'o-', label='Theorem 1.1 upper bound (Eq. (26))')
    plt.plot(n_array, lower_treewidth, 'o-', label=f'Treewidth lower bound (Eq. (23))')
    plt.plot(n_array, upper_treewidth, 'o-', label=f'Treewidth upper bound (Eq. (23))')
    plt.xlabel('$n$')
    plt.title('cng(' + (f'$P_{m}$' + u'\u25A1' + '$P_n$' if not p else f'$C_{m}$' + u'\u25A1' + '$C_n$') + ')')
    plt.legend(loc='upper left')
    plt.savefig(f'output/lattice-m{m}-periodic{p}' + '.pdf', format='pdf', bbox_inches='tight')
    plt.show()


# EXPERIMENT 2: SQUARE LATTICE
k_array = list(range(5, 21))
for p in [True, False]:
    plt.figure(1, figsize=(5,3))
    plt.ylim(-50, 600)
    lower_main = [lower_bd_main_thm(k, k, p) for k in k_array]
    upper_main = [upper_bd_main_thm(k, k, p) for k in k_array]
    upper_treewidth = [upper_bd_treewidth(k, k, p) for k in k_array]
    lower_treewidth = [(i-4)/4 for i in upper_treewidth]
    print(f'Lower main:{lower_main}\nUpper main:{upper_main}\nUpper tw:{upper_treewidth}')
    plt.plot(n_array, lower_main, 'o-', label='Theorem 1.1 lower bound (Eq. (25))')
    plt.plot(n_array, upper_main, 'o-', label='Theorem 1.1 upper bound (Eq. (26))')
    plt.plot(n_array, lower_treewidth, 'o-', label=f'Treewidth lower bound (Eq. (23))')
    plt.plot(n_array, upper_treewidth, 'o-', label=f'Treewidth upper bound (Eq. (23))')
    plt.xlabel('$k$')
    plt.title('cng(' + ('$P_k$' + u'\u25A1' + '$P_k$' if not p else '$C_k$' + u'\u25A1' + '$C_k$') + ')')
    plt.legend(loc='upper left')
    plt.savefig(f'output/square-lattice-periodic{p}' + '.pdf', format='pdf', bbox_inches='tight')
    plt.show()
