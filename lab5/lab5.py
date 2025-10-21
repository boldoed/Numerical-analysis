# вариант 14 (a(ij) = 1/cosh(i+j))
# степенной метод

import numpy as np
import math

def make_matrix(n):
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            A[i, j] = 1/math.cosh(i+j+2)
    return A

# Степенной метод вычисления максимального по модулю собственного числа.
def power_method(A, eps=1e-10, max_iter=10**6):
    n, m = A.shape
    if n != m:
        raise ValueError("A не квадратная")
    y_prev = np.random.rand(n)
    x0 = y_prev
    y_prev /= np.linalg.norm(y_prev)
    for _ in range(max_iter):
        y_new = A @ y_prev
        y_new /= np.linalg.norm(y_new)
        if np.linalg.norm(y_new - y_prev) < eps:
            break
        y_prev = y_new

    max_eigen = np.linalg.norm(A @ y_new)
    max_vector = y_new

    return max_eigen, max_vector, x0

def inverse_power_method(A, eps=1e-10, max_iter=1000):
    n, m = A.shape
    if n != m:
        raise ValueError("A не квадратная")
    y_prev = np.random.rand(n)
    y_prev /= np.linalg.norm(y_prev)
    for _ in range(max_iter):
        # Вместо A_inv @ y /|\ A @ y_new = y
        y_new = np.linalg.solve(A, y_prev)
        y_new /= np.linalg.norm(y_new)
        if np.linalg.norm(y_new - y_prev) < eps:
            break
        y_prev = y_new     
    min_eigen = np.linalg.norm(A @ y_prev)
    min_vector = y_new

    return min_eigen, min_vector

def max_pos(A):
    n = A.shape[0]
    max_val = -1.0
    pos = (0, 0)
    for i in range(n):
        for j in range(i+1, n):
            if abs(A[i, j]) > max_val:
                max_val = abs(A[i, j])
                pos = (i, j)
    return pos

def jacobi_method(A, eps=10e-10, max_iterations=10**6):
    n, m = A.shape
    if n != m:
        raise ValueError("A не квадратная")
    iterations = 0
    Q = np.eye(n)

    while (np.sum(A**2) - np.sum(np.diag(A)**2) >= eps) and (iterations <= max_iterations):
        i, j = max_pos(A)
        if A[i, i] != A[j, j]:
            angle = np.arctan(2 * A[i, j] / (A[i, i] - A[j, j])) / 2
        else:
            angle = np.pi / 4
        P = np.eye(n)
        P[i, i] = P[j, j] = np.cos(angle)
        P[i, j] = -np.sin(angle)
        P[j, i] = np.sin(angle)
        A = P.T @ A @ P
        Q = Q @ P
        iterations += 1

    eigen_values = np.diag(A)
    eigen_vectors = Q

    return (eigen_values, eigen_vectors)

if __name__ == "__main__":
    try:
        A = make_matrix(4)
        max_eigen, max_vector, x0 = power_method(A)
        # A_inv = np.linalg.inv(A)
        # min_eigen, min_vector = power_method(A_inv)
        # min_eigen = 1/min_eigen
        min_eigen, min_vector = inverse_power_method(A)
        r = A @ max_vector - max_eigen * max_vector
        print("матрица A:")
        print(A)
        print(f"max λ: {max_eigen}")
        print(f"вектор:{max_vector}")
        print(f"невязка:{r}")
        print(f"x0:{x0}")
        # проверка через numpy
        # w = np.linalg.eigvals(A)
        # w = np.abs(w)
        # print("numpy λ_min =", np.min(w))
        # print("numpy λ_max =", np.max(w))
        values, vectors = jacobi_method(A)
        for i, v in enumerate(values):
            vector = vectors[i]
            print(f"{i}     {v}")
            print(f"    {vector}")
            print(f"    {A @ vector - v * vector}")

    except Exception as e:
        print(f"ошибка: {e}")
