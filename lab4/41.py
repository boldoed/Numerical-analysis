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


def jacobi(A, b, eps=1e-10, max_iter=1000):
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

def steepest_descent(A, b, eps=1e-10, max_iter=1000):
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
        Ar = A @ r
        r_r = r @ r # r^T r
        r_Ar = r @ (A @ r) # r^T Ar
        if r_Ar == 0:
            raise ValueError("r^T Ar = 0")
        tau = r_r / r_Ar

        x_prev = x_cur
        x_cur = x_prev + tau * r

    return x_cur, np.array(losses)

if __name__ == "__main__":
    try:
        matrix_1 = read_matrix("lab4_1.txt")

        correct_x = [1, 2, 3]
        full_matr_1 = create_full_matrix(matrix_1, correct_x)
        for i in range(len(full_matr_1)):
            print(full_matr_1[i])
        # full_hilbert = hilbert
        # for i in range(len(hilbert)):
        #     full_hilbert[i].append(b_hilbert[i])   
        
    except Exception as e:
        print(f"ошибка: {e}")


# x=D^(−1) * (b − (L + U) * x)