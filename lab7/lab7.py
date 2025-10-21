import math
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 1.0 / (1.0 + 25.0 * x * x)

def g(x):
    return (math.sin(2 * x) + math.sin(3 * x) ** 2) / (3.0 + math.sin(x) + math.cos(2 * x))

#метод прогонки для трёхдиагональной системы
def tridiagonal_matrix(A, b):
    n = A.shape[0]
    x = np.zeros(n)
    alpha = np.zeros(n - 1)
    beta = np.zeros(n)

    # прямой ход
    alpha[0] = -A[0, 1] / A[0, 0]
    beta[0] = b[0] / A[0, 0]
    for i in range(1, n - 1):
        denom = A[i, i] + A[i, i - 1] * alpha[i - 1]
        alpha[i] = -A[i, i + 1] / denom
        beta[i] = (b[i] - A[i, i - 1] * beta[i - 1]) / denom
    beta[n - 1] = (b[n - 1] - A[n - 1, n - 2] * beta[n - 2]) / (A[n - 1, n - 1] + A[n - 1, n - 2] * alpha[n - 2])

    # обратный ход
    x[n - 1] = beta[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]

    return x

# полином Лагранжа
def lagrange(f, x1, x2, n):
    x_nodes = np.linspace(x1, x2, n)
    f_vals = np.array([f(x_i) for x_i in x_nodes])
    def _lagrange(x):
        L = 0.0
        for k in range(n):
            term = f_vals[k]
            for i in range(n):
                if i == k:
                    continue
                term *= (x - x_nodes[i]) / (x_nodes[k] - x_nodes[i])
            L += term
        return L

    return _lagrange


# вторые производные для сплайна
def find_c(h, f_vals, n):
    if n <= 2:
        return np.zeros(n)
    A = np.zeros((n - 2, n - 2))
    b = np.zeros(n - 2)
    for i in range(1, n - 1):
        idx = i - 1
        if i > 1:
            A[idx, idx - 1] = h[i - 1]  # нижняя диагональ
        A[idx, idx] = 2.0 * (h[i - 1] + h[i])  # главная диагональ
        if i < n - 2:
            A[idx, idx + 1] = h[i]  # верхняя диагональ    
        b[idx] = 3.0 * ((f_vals[i + 1] - f_vals[i]) / h[i] - (f_vals[i] - f_vals[i - 1]) / h[i - 1])
    c_inner = tridiagonal_matrix(A, b)
    c = np.zeros(n)
    c[1:n - 1] = c_inner
    return c


# кубический сплайн
def cubic_spline(f, x1, x2, n):
    x_nodes = np.linspace(x1, x2, n)
    f_vals = np.array([f(x) for x in x_nodes])
    h = np.diff(x_nodes)  # длины отрезков
    c = find_c(h, f_vals, n)
    a = f_vals[:-1]  # a_i = f(x_i)
    b = np.zeros(n - 1)
    d = np.zeros(n - 1)
    for i in range(n - 1):
        b[i] = (f_vals[i + 1] - f_vals[i]) / h[i] - h[i] * (c[i + 1] + 2.0 * c[i]) / 3.0
        d[i] = (c[i + 1] - c[i]) / (3.0 * h[i])
    def _cubic_spline(x):
        # Найти индекс интервала [x_i, x_i+1]
        i = np.searchsorted(x_nodes, x) - 1
        i = np.clip(i, 0, n - 2)
        dx = x - x_nodes[i]
        return a[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx

    return _cubic_spline


if __name__ == "__main__":
    print("Интерполяция f(x)")
    x5 = np.linspace(-1, 1, 5)
    x20 = np.linspace(-1, 1, 20)
    lagrange_f5 = lagrange(f, -1, 1, 5)
    lagrange_f20 = lagrange(f, -1, 1, 20)
    print("значения между узлами")
    print(f"N=5:  слева = {lagrange_f5((x5[0] + x5[1]) / 2)}, справа = {lagrange_f5((x5[-2] + x5[-1]) / 2)}")
    print(f"N=20: слева = {lagrange_f20((x20[0] + x20[1]) / 2)}, справа = {lagrange_f20((x20[-2] + x20[-1]) / 2)}")

    # график для Лагранжа
    x_plot = np.linspace(-1, 1, 400)
    y_true = np.array([f(x) for x in x_plot])
    y_l5 = np.array([lagrange_f5(x) for x in x_plot])
    y_l20 = np.array([lagrange_f20(x) for x in x_plot])

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_true, 'g-', label='f(x)', linewidth=2)
    plt.plot(x_plot, y_l5, 'b-', label='Lagrange N=5')
    plt.plot(x_plot, y_l20, 'r-', label='Lagrange N=20')
    plt.legend()
    plt.title('интерполяция полиномом Лагранжа')
    plt.show()

    # сплайны
    spline_f5 = cubic_spline(f, -1, 1, 5)
    spline_f20 = cubic_spline(f, -1, 1, 20)

    print("значения кубического сплайна между узлами")
    print(f"N=5:  слева = {spline_f5((x5[0] + x5[1]) / 2)}, справа = {spline_f5((x5[-2] + x5[-1]) / 2)}")
    print(f"N=20: слева = {spline_f20((x20[0] + x20[1]) / 2)}, справа = {spline_f20((x20[-2] + x20[-1]) / 2)}")

    y_s5 = np.array([spline_f5(x) for x in x_plot])
    y_s20 = np.array([spline_f20(x) for x in x_plot])

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_true, 'g-', label='f(x)', linewidth=2)
    plt.plot(x_plot, y_s5, 'b-', label='Spline N=5')
    plt.plot(x_plot, y_s20, 'r-', label='Spline N=20')
    plt.title('интерполяция кубическим сплайном')
    plt.legend()
    plt.show()
    print()
    print("Интерполяция g(x)")
    
    lagrange_g20 = lagrange(g, 0, math.pi, 20)
    spline_g20 = cubic_spline(g, 0, math.pi, 20)
    x_plot_g = np.linspace(0, math.pi, 400)
    y_true_g = np.array([g(x) for x in x_plot_g])
    y_l_g = np.array([lagrange_g20(x) for x in x_plot_g])
    y_s_g = np.array([spline_g20(x) for x in x_plot_g])

    # вычисление значений посередине между всеми узлами
    x_nodes_g = np.linspace(0, math.pi, 20)
    midpoints = [(x_nodes_g[i] + x_nodes_g[i+1]) / 2 for i in range(len(x_nodes_g) - 1)]
    lagrange_vals = [lagrange_g20(x) for x in midpoints]
    spline_vals = [spline_g20(x) for x in midpoints]

    for i in range(len(midpoints)):
        print(f"  x = {midpoints[i]} L = {lagrange_vals[i]}, S = {spline_vals[i]}")

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot_g, y_true_g, 'g-', label='g(x)', linewidth=2)
    plt.plot(x_plot_g, y_l_g, 'b-', label='Lagrange N=20')
    plt.plot(x_plot_g, y_s_g, 'r-', label='Spline N=20')
    plt.legend()
    plt.title('Интерполяция g(x)')
    plt.show()