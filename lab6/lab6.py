import math
import numpy as np
import matplotlib.pyplot as plt

def f_safe(x):
    if x < math.log(2) or x > math.log(4):
        return float('nan')
    y = math.exp(x) - 3
    return math.acos(y) - x

def F(point):
    x, y = point
    f1 = math.sin(x - 1) + y - 1.5
    f2 = x - math.sin(y + 1) - 1
    return np.array([f1, f2])

def J(point):
    x, y = point
    j11 = math.cos(x - 1)
    j12 = 1.0
    j21 = 1.0
    j22 = -math.cos(y + 1)
    return np.array([[j11, j12],
                     [j21, j22]])

def df_safe(x):
    if x <= math.log(2) or x >= math.log(4):
        return float('nan')
    exp_x = math.exp(x)
    denom = math.sqrt(1 - (exp_x - 3)**2)
    return -exp_x / denom - 1

# Метод простых итераций 
def simple_iteration(phi, x0, eps=1e-10, max_iter=1000):
    x_min = math.log(2)
    x_max = math.log(4)
    if not (x_min < x0 < x_max):
        raise ValueError(f"x0={x0} вне допустимого интервала")
    
    x_prev = x0
    for k in range(1, max_iter + 1):
        try:
            x_new = phi(x_prev)
        except Exception as e:
            raise ValueError(f"ошибка в phi({x_prev}): {e}")
        # защита от выхода за границы
        if x_new <= x_min:
            x_new = x_min + 1e-12
        elif x_new >= x_max:
            x_new = x_max - 1e-12
    
        if abs(x_new - x_prev) < eps:
            return x_new, k
        x_prev = x_new
    raise RuntimeError("Не сошёлся")

# Метод Ньютона
def newton_method(f, df, x0, eps=1e-12, max_iter=100):
    x = x0
    values = []
    for _ in range(max_iter):
        fx = f(x)
        values.append(fx)
        if abs(fx) < eps:
            return values, x
        dfx = df(x)
        if dfx == 0 or math.isnan(dfx):
            raise ValueError("Производная нулевая или не определена")
        x = x - fx / dfx
        if math.isnan(x):
            raise ValueError("x стал nan")
    raise RuntimeError("Не сошёлся")

# Метод Ньютона для системы
def newton_system(F, J, x0, eps=1e-10, max_iter=100):
    x = np.array(x0)
    values = []  # сохраняем норму невязки на каждой итерации
    for k in range(max_iter):
        F_val = F(x)
        norm_F = np.linalg.norm(F_val)
        values.append(norm_F) 
        if norm_F < eps:
            return x, values, k + 1
        J_mat = J(x)
        try:
            dx = np.linalg.solve(J_mat, -F_val)
        except:
            print("матрица Якоби вырождена")
        
        x = x + dx
    
    raise RuntimeError(f"не сошёлся за {max_iter} итераций")

alpha = 0.0001
def phi_direct(x):
    return math.acos(math.exp(x) - 3)

def phi_relax(x):
    return x - alpha * f_safe(x)


x_min = math.log(2)
x_max = math.log(4)
x_vals = np.linspace(x_min + 1e-8, x_max - 1e-8, 500)
y_vals = [f_safe(x) for x in x_vals]

plt.figure(figsize=(10, 5))
plt.plot(x_vals, y_vals, 'b-', linewidth=2)
plt.axhline(0, color='k', linestyle='--')
plt.show()


if __name__ == "__main__":
    x0 = 1 
    # метод простых итераций сходится, только если ∣φ′(x)∣<1 в окрестности корня.
    print("метод простых итераций")
    try:
        root1, it1 = simple_iteration(phi_direct, x0)
        print(f"Прямой метод: x = {root1}, итераций = {it1}, f(x) = {f_safe(root1)}")
    except Exception as e:
        print("Прямой метод не сработал:", e)
    
    try:
        root2, it2 = simple_iteration(phi_relax, x0)
        print(f"Релаксационный (α={alpha}): x = {root2}, итераций = {it2}, f(x) = {f_safe(root2)}")
    except Exception as e:
        print("Релаксационный метод не сработал:", e)
    
    print("Метод Ньютона")
    try:
        vals, root_newton = newton_method(f_safe, df_safe, x0)
        print(f"Ньютон: x = {root_newton}, f(x) = {f_safe(root_newton)}, итераций = {len(vals)}")
    except Exception as e:
        print("Метод Ньютона не сработал:", e)

    x0 = [1.0, 1.0]
    
    try:
        solution, residuals, iterations = newton_system(F, J, x0)
        print(f"Решение: x = {solution[0]}, y = {solution[1]}")
        print(f"итераций: {iterations}")
        print(f"Проверка F(x,y): {F(solution)}")

    except Exception as e:
        print("Ошибка:", e)