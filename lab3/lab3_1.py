import math 

def read_matrix(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    matrix = []
    for line in lines:
        row = [float(x) for x in line.strip().split()]
        matrix.append(row)
    return matrix

def calculate_residual(original_matrix, solution=4):
    n = len(original_matrix)
    residual = []
    for i in range(n):
        lhs = sum((original_matrix[i][j] * solution[j]) 
               for j in range(n))
        rhs = original_matrix[i][n]
        residual.append((lhs - rhs))
    return residual

# метод оптимального исключения
def optimal_exclusion(matrix=4):
    n = len(matrix)
    
    for i in range(n):
        max_row = i
        for k in range(i, n):
            if abs(matrix[k][i]) > abs(matrix[max_row][i]):
                max_row = k
        
        matrix[i], matrix[max_row] = matrix[max_row], matrix[i]
        
        pivot = matrix[i][i]
        if pivot == 0:
            raise ValueError("система вырождена")
        
        for j in range(i, n + 1):
            matrix[i][j] = (matrix[i][j] / pivot)
        
        # Исключение переменной
        for k in range(n):
            if k != i and matrix[k][i] != 0:
                factor = (matrix[k][i])
                for j in range(i, n + 1):
                    matrix[k][j] = (matrix[k][j] - factor * matrix[i][j])
    
    solution = [(matrix[i][n]) for i in range(n)] 

    return solution, matrix

# метод вращений 
def rotation_method(matrix=4):
    n = len(matrix)
    mat = [row.copy() for row in matrix]
    
    Ab = [row.copy() for row in mat]
    
    for i in range(n):
        for j in range(i+1, n):
            if Ab[j][i] == 0:
                continue
                
            norm = math.sqrt(Ab[i][i] ** 2 + Ab[j][i] ** 2)
            cos = (Ab[i][i] / norm)
            sin = (Ab[j][i] / norm)
            
            row_i = [(cos * Ab[i][k] + sin * Ab[j][k]) 
                    for k in range(n+1)]
            row_j = [(-sin * Ab[i][k] + cos * Ab[j][k]) 
                    for k in range(n+1)]
            
            Ab[i] = row_i
            Ab[j] = row_j

    solution = [0] * n
    for i in range(n-1, -1, -1):
        sum_ax = sum((Ab[i][j] * solution[j]) 
                for j in range(i+1, n))
        solution[i] = ((Ab[i][n] - sum_ax) / Ab[i][i])
    
    return solution

def hilbert_matrix(n):
    matrix = []
    for i in range(1, n+1):
        row = []
        for j in range(1, n + 1):
            row.append(1 / (i + j - 1))
        matrix.append(row)
    return matrix

def matrix_vector_multiply(matrix, x):
    n = len(matrix)
    result = [0] * n  
    for i in range(n):
        for j in range(n):
            result[i] += matrix[i][j] * x[j]
    return result


def inverse_matrix(matrix):
    "метод Гаусса-Жордана"
    n = len(matrix)
    augmented = [row[:] + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(matrix)]
    for i in range(n):
        max_row = i
        for k in range(i, n):
            if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                max_row = k
        augmented[i], augmented[max_row] = augmented[max_row], augmented[i]
        
        pivot = augmented[i][i]
        if abs(pivot) < 1e-10:
            raise ValueError("Матрица вырождена")
        
        for j in range(i, 2*n):
            augmented[i][j] /= pivot
        
        for k in range(n):
            if k != i:
                factor = augmented[k][i]
                for j in range(i, 2*n):
                    augmented[k][j] -= factor * augmented[i][j]

    inverse = [[augmented[i][j] for j in range(n, 2*n)] for i in range(n)]
    return inverse

def matrix_norm_1(matrix):
    n = len(matrix)
    column_sums = [sum(abs(matrix[i][j]) for i in range(n)) for j in range(n)]
    return max(column_sums)

def condition_number(matrix):
    norm_A = matrix_norm_1(matrix)
    try:
        A_inv = inverse_matrix(matrix)
    except ValueError as e:
        raise ValueError("Матрица вырождена, число обусловленности бесконечно")
    norm_A_inv = matrix_norm_1(A_inv)
    return norm_A * norm_A_inv

if __name__ == "__main__":
    try:
        matrix = read_matrix("lab2.txt")

        hilbert = hilbert_matrix(4)
        correct_ans = [1, 2, 3, 4]
        b_hilbert = matrix_vector_multiply(hilbert, correct_ans)
        full_hilbert = hilbert
        for i in range(len(hilbert)):
            full_hilbert[i].append(b_hilbert[i])
            # print(hilbert[i], b_hilbert[i])    

        # Решение методом оптимального исключения
        solution_opt, matr = optimal_exclusion([row.copy() for row in full_hilbert])
        residual_opt = calculate_residual(full_hilbert, solution_opt)
        
        # Решение методом вращений
        solution_rot = rotation_method([row.copy() for row in full_hilbert])
        residual_rot = calculate_residual(full_hilbert, solution_rot)

        print("метод оптимального исключения:")
        for i, x in enumerate(solution_opt, 1):
            print(f"x{i} = {x}")
        print(f"вектор невязки: {residual_opt}\n")
        
        print("метод вращений:")
        for i, x in enumerate(solution_rot, 1):
            print(f"x{i} = {x}")
        print(f"вектор невязки: {residual_rot}")
        
        cond_hilbert = condition_number(hilbert)
        print(f"Число обусловленности матрицы Гильберта: {cond_hilbert}")

        # matr = [[matrix[i][j] for j in range(len(matrix) - 1)] for i in range(len(matrix))]
        cond_matr = condition_number(matrix)
        print(f"Число обусловленности матрицы из 2 лабы: {cond_matr}")     
        

    except Exception as e:
        print(f"ошибка: {e}")