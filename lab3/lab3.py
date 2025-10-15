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

# метод оптимельного исключения
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
    matr = []
    for i in range(n):
        row = []  # Создаем новый список для каждой строки
        for j in range(n):
            row.append(1 / (i + j - 1))
        matr.append(row)
    return matr

if __name__ == "__main__":
    try:
        matrix = read_matrix("lab2.txt")
        n = 4
        hilbert = hilbert_matrix(n)
        x_true = [1.0] * n
        b = [sum(hilbert[i][j] * x_true[j] for j in range(n)) for i in range(n)]
        
        hilbert_full = [hilbert[i] + [b[i]] for i in range(n)]

        # Решение методом оптимального исключения
        solution_opt, matr = optimal_exclusion([row.copy() for row in hilbert_full])
        residual_opt = calculate_residual(hilbert_full, solution_opt)
        
        # Решение методом вращений
        solution_rot = rotation_method([row.copy() for row in hilbert_full])
        residual_rot = calculate_residual(hilbert_full, solution_rot)

        print("метод оптимального исключения:")
        for i, x in enumerate(solution_opt, 1):
            print(f"x{i} = {x}")
        print(f"вектор невязки: {residual_opt}\n")
        
        print("метод вращений:")
        for i, x in enumerate(solution_rot, 1):
            print(f"x{i} = {x}")
        print(f"вектор невязки: {residual_rot}")

        # for i in range(4):
        #     print(matr[i])
        
        # for i in range(4):
        #     print(hilbert[i])
        
    except Exception as e:
        print(f"ошибка: {e}")