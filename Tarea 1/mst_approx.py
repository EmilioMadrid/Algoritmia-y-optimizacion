import numpy as np
import matplotlib.pyplot as plt
import itertools
import random

# Semillas para reproducibilidad
np.random.seed(42)
random.seed(42)

def total_distance(tour, dist_matrix):
    return sum(dist_matrix[tour[i], tour[(i+1)%len(tour)]] for i in range(len(tour)))

def exact_tsp(dist_matrix):
    # Fuerza bruta para encontrar el óptimo exacto (viable solo para n pequeño, ej. 9)
    n = len(dist_matrix)
    best_cost = float('inf')
    for perm in itertools.permutations(range(1, n)):
        tour = [0] + list(perm)
        cost = total_distance(tour, dist_matrix)
        if cost < best_cost:
            best_cost = cost
    return best_cost

def mst_approximation(dist_matrix):
    n = len(dist_matrix)
    selected = [False]*n
    selected[0] = True
    mst = np.zeros((n, n))
    edges = 0
    
    # Algoritmo de Prim para el MST
    while edges < n - 1:
        min_edge = float('inf')
        u, v = -1, -1
        for i in range(n):
            if selected[i]:
                for j in range(n):
                    if not selected[j] and dist_matrix[i, j] < min_edge:
                        min_edge = dist_matrix[i, j]
                        u, v = i, j
        if u != -1 and v != -1:
            mst[u, v] = min_edge
            mst[v, u] = min_edge
            selected[v] = True
            edges += 1
            
    # Recorrido en preorden (DFS) para generar el tour
    visited = [False]*n
    tour = []
    def dfs(node):
        visited[node] = True
        tour.append(node)
        for neighbor in range(n):
            if mst[node, neighbor] > 0 and not visited[neighbor]:
                dfs(neighbor)
    dfs(0)
    return tour, total_distance(tour, dist_matrix)

def two_opt(initial_tour, dist_matrix):
    tour = list(initial_tour)
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n if i > 0 else n - 1):
                a, b = tour[i], tour[i+1]
                c, d = tour[j], tour[(j+1)%n]
                delta = (dist_matrix[a, c] + dist_matrix[b, d]) - (dist_matrix[a, b] + dist_matrix[c, d])
                if delta < -1e-6:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
                    break
            if improved:
                break
    return tour, total_distance(tour, dist_matrix)

def simulated_annealing(dist_matrix):
    n = len(dist_matrix)
    current_tour = list(range(n))
    random.shuffle(current_tour)
    current_cost = total_distance(current_tour, dist_matrix)
    best_cost = current_cost
    T = 100
    T_min = 0.1
    param = 0.95
    max_iter = 30
    
    while T > T_min:
        for _ in range(max_iter):
            i, j = sorted(random.sample(range(n), 2))
            neighbor = current_tour[:]
            neighbor[i:j+1] = list(reversed(current_tour[i:j+1]))
            neighbor_cost = total_distance(neighbor, dist_matrix)
            delta = neighbor_cost - current_cost
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_tour = neighbor
                current_cost = neighbor_cost
                if current_cost < best_cost:
                    best_cost = current_cost
        T *= param
    return best_cost

if __name__ == '__main__':
    num_instances = 100
    ratios = []
    mst_costs, mst_2opt_costs, ls_costs, sa_costs = [], [], [], []

    print(f"Evaluando {num_instances} instancias de 9 ciudades... (Esto puede tardar unos segundos)")
    
    for _ in range(num_instances):
        # Generar ciudades y matriz de distancias
        cities = np.random.rand(9, 2) * 100
        diff = cities[:, np.newaxis, :] - cities[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
        
        # 1. Óptimo Exacto (Fuerza Bruta)
        c_opt = exact_tsp(dist_matrix)
        
        # 2. MST Puro
        tour_mst, c_mst = mst_approximation(dist_matrix)
        
        # 3. MST + 2-opt
        _, c_mst_2opt = two_opt(tour_mst, dist_matrix)
        
        # 4. Local Search (Aleatorio + 2-opt)
        rand_tour = list(range(9))
        random.shuffle(rand_tour)
        _, c_ls = two_opt(rand_tour, dist_matrix)
        
        # 5. Simulated Annealing
        c_sa = simulated_annealing(dist_matrix)
        
        # Almacenar resultados
        ratios.append(c_mst / c_opt)
        mst_costs.append(c_mst)
        mst_2opt_costs.append(c_mst_2opt)
        ls_costs.append(c_ls)
        sa_costs.append(c_sa)

    # Generación de gráficas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.hist(ratios, bins=15, color='purple', alpha=0.7, edgecolor='black')
    ax1.axvline(x=2, color='red', linestyle='dashed', linewidth=2, label='Límite Teórico (2)')
    ax1.set_title('Razón de Aproximación: MST vs Óptimo')
    ax1.set_xlabel('Costo MST / Costo Óptimo')
    ax1.set_ylabel('Frecuencia (Instancias)')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.7)

    # Implementado con tick_labels para compatibilidad
    ax2.boxplot([mst_costs, mst_2opt_costs, ls_costs, sa_costs], tick_labels=['MST Puro', 'MST + 2-opt', 'Local Search\n(Rand)', 'Simulated\nAnnealing'])
    ax2.set_title('Comparación de Métodos (Costo Obtenido)')
    ax2.set_ylabel('Costo Total')
    ax2.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    
    print(f"\nResultados promedio ({num_instances} instancias):")
    print(f"Razón media (MST/Óptimo): {np.mean(ratios):.3f} (Máximo: {np.max(ratios):.3f})")
    print(f"MST Puro: {np.mean(mst_costs):.1f}")
    print(f"MST + 2-opt: {np.mean(mst_2opt_costs):.1f}")
    print(f"Local Search (Rand): {np.mean(ls_costs):.1f}")
    print(f"Simulated Annealing: {np.mean(sa_costs):.1f}")
    
    plt.show()