import math 
import numpy as np

def read_matrix(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    matrix = []
    for line in lines:
        row = [float(x) for x in line.strip().split()]
        matrix.append(row)
    return matrix

# метод Якоби сходится, если матрица строго диагонально доминируемая
def jacobi(A, b, eps=1e-10, max_iter=20):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("A не квадратная")
    B = np.zeros((n, n))
    g = np.zeros(n)
    for i in range(n):
        if A[i, i] == 0:
            raise ValueError(f"элемент A[{i},{i}] равен нулю")
        for j in range(n):
            if i != j:
                B[i, j] = -A[i, j] / A[i, i]
        g[i] = b[i] / A[i, i]  
    # максимальная сумма модулей по строкам
    B_norm = np.max(np.sum(np.abs(B), axis=1)) 
    if B_norm >= 1:
        print("||B|| >= 1")
    x_prev = np.zeros(n)
    x_cur = np.zeros(n)
    losses = [] # норма невязки
    for _ in range(max_iter):
        residual = np.linalg.norm(A @ x_cur - b)
        losses.append(residual)
        diff = np.linalg.norm(x_cur - x_prev)
        if B_norm < 1 and (B_norm * diff / (1.0 - B_norm) < eps):
            break
        x_prev = x_cur
        x_cur = B @ x_prev + g
    else:
        print("не сошлось((")
    
    return x_cur, np.array(losses)

# метод скорейшего спуска сходится, если матрица симметричная положительно определённая
def steepest_descent(A, b, eps=1e-10, max_iter=20):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("A не квадратная")
    x_prev = np.zeros(n)
    x_cur = np.zeros(n)
    losses = []

    for _ in range(max_iter):
        r = b - A @ x_cur # невязкa
        residual_norm = np.linalg.norm(r)
        losses.append(residual_norm)
        if residual_norm < eps:
            break
        r_r = r @ r # r^T r
        r_Ar = r @ (A @ r) # r^T Ar
        if r_Ar == 0:
            raise ValueError("r^T Ar = 0")
        tau = r_r / r_Ar

        x_prev = x_cur
        x_cur = x_prev + tau * r

    return x_cur, np.array(losses)

def create_full_matrix(A, x):
    A = np.array(A)
    x = np.array(x)
    n, m = A.shape
    if n != m:
        raise ValueError("A не квадратная")
    if x.shape[0] != n:
        raise ValueError("вектор x не совместим с A.")
    b = A @ x
    return b

def create_T_sle(A, b):
    A_new = A.transpose() @ A
    b_new = A.transpose() @ b
    return A_new, b_new

if __name__ == "__main__":
    try:
        A1 = read_matrix("lab4_1.txt")
        x1 = [1, 2, 3]
        b1 = create_full_matrix(A1, x1)
        print("Якоби")
        x_jacobi, losses_jacobi = jacobi(A1, b1)
        r_jacobi = A1 @ x_jacobi - b1
        print(x_jacobi, r_jacobi)
        print()
        print("скорейшего спуска")
        x_steepest_descent, losses_steepest_descent = steepest_descent(A1, b1)
        r_steepest_descent = A1 @ x_steepest_descent - b1
        print(x_steepest_descent, r_steepest_descent)
        

        print()
        print()

        A2 = read_matrix("lab4_2.txt")
        x2 = [1, 2, 3, 4]
        b2 = create_full_matrix(A2, x2)
        print("Якоби")
        x_jacobi, losses_jacobi = jacobi(A2, b2)
        r_jacobi = A2 @ x_jacobi - b2
        print(x_jacobi, r_jacobi)
        print()
        print("скорейшего спуска")
        x_steepest_descent, losses_steepest_descent = steepest_descent(A2, b2)
        r_steepest_descent = A2 @ x_steepest_descent - b2
        print(x_steepest_descent, r_steepest_descent)

        print()
        print()
        A1 = np.array(A1)
        b1 = np.array(b1)
        At, bt = create_T_sle(A1, b1)
        print("Якоби")
        x_jacobi, losses_jacobi = jacobi(At, bt)
        r_jacobi = At @ x_jacobi - bt
        print(x_jacobi, r_jacobi)
        print()
        print("скорейшего спуска")
        x_steepest_descent, losses_steepest_descent = steepest_descent(At, bt)
        r_steepest_descent = At @ x_steepest_descent - bt
        print(x_steepest_descent, r_steepest_descent)        


    except Exception as e:
        print(f"ошибка: {e}")


# x=D^(−1) * (b − (L + U) * x)