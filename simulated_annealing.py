import numpy as np
import random
import time
import matplotlib.pyplot as plt

# Semillas para reproducibilidad
np.random.seed(42)
random.seed(42)

# Generación de la instancia: 50 ciudades
num_cities = 50
cities = np.random.rand(num_cities, 2) * 100

def distance_matrix(cities):
    diff = cities[:, np.newaxis, :] - cities[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))

dist_matrix = distance_matrix(cities)

def total_distance(tour, dist_matrix):
    return sum(dist_matrix[tour[i], tour[(i+1)%len(tour)]] for i in range(len(tour)))

def get_neighbor(tour):
    # Operador 2-opt: invierte una subsecuencia al azar
    n = len(tour)
    i, j = sorted(random.sample(range(n), 2))
    new_tour = tour[:]
    new_tour[i:j+1] = list(reversed(tour[i:j+1]))
    return new_tour

def simulated_annealing(dist_matrix, schedule, T0=100, T_min=0.1, param=0.99, max_iter=50):
    n = len(dist_matrix)
    current_tour = list(range(n))
    random.shuffle(current_tour)
    current_cost = total_distance(current_tour, dist_matrix)
    
    best_tour = current_tour[:]
    best_cost = current_cost
    
    T = T0
    history_cost = []
    start_time = time.time()
    
    while T > T_min:
        for _ in range(max_iter):
            neighbor = get_neighbor(current_tour)
            neighbor_cost = total_distance(neighbor, dist_matrix)
            delta = neighbor_cost - current_cost
            
            # Criterio de aceptación de Metropolis
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_tour = neighbor
                current_cost = neighbor_cost
                
                if current_cost < best_cost:
                    best_tour = current_tour[:]
                    best_cost = current_cost
            
            history_cost.append(current_cost)
        
        # Enfriamiento
        if schedule == 'exponential':
            T *= param
        elif schedule == 'linear':
            T -= param
            
    exec_time = time.time() - start_time
    return best_tour, best_cost, history_cost, exec_time

if __name__ == '__main__':
    T0 = 100
    T_min = 0.1
    max_iter = 50

    print("Ejecutando simulación inicial para curvas de convergencia...")
    # Calculamos los parámetros (0.95 exponencial y 0.74 lineal) para que ambos realicen ~134 ciclos de temperatura
    _, _, hist_exp, _ = simulated_annealing(dist_matrix, 'exponential', T0=T0, T_min=T_min, param=0.95, max_iter=max_iter)
    _, _, hist_lin, _ = simulated_annealing(dist_matrix, 'linear', T0=T0, T_min=T_min, param=0.74, max_iter=max_iter)

    trials = 20
    exp_times, exp_costs = [], []
    lin_times, lin_costs = [], []

    print(f"Ejecutando {trials} pruebas comparativas. Esto tomará unos segundos...")
    for _ in range(trials):
        _, cost_e, _, time_e = simulated_annealing(dist_matrix, 'exponential', T0=T0, T_min=T_min, param=0.95, max_iter=max_iter)
        exp_times.append(time_e)
        exp_costs.append(cost_e)
        
        _, cost_l, _, time_l = simulated_annealing(dist_matrix, 'linear', T0=T0, T_min=T_min, param=0.74, max_iter=max_iter)
        lin_times.append(time_l)
        lin_costs.append(cost_l)

    # Generación de gráficas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(hist_exp, label='Exponencial ($\\alpha=0.95$)', color='blue')
    ax1.plot(hist_lin, label='Lineal ($\\Delta T=0.74$)', color='red', alpha=0.7)
    ax1.set_title('Convergencia: Costo vs. Iteraciones')
    ax1.set_xlabel('Iteraciones (Llamadas a la función objetivo)')
    ax1.set_ylabel('Costo (Distancia Total)')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.7)

    ax2.scatter(exp_times, exp_costs, label='Exponencial', color='blue', alpha=0.7)
    ax2.scatter(lin_times, lin_costs, label='Lineal', color='red', alpha=0.7)
    ax2.set_title(f'Tiempo de Ejecución vs Calidad ({trials} pruebas)')
    ax2.set_xlabel('Tiempo de Ejecución (segundos)')
    ax2.set_ylabel('Distancia Final (menor es mejor)')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    
    print(f"\nResultados promedio ({trials} pruebas):")
    print(f"Exponencial - Costo: {np.mean(exp_costs):.1f}, Tiempo: {np.mean(exp_times):.4f}s")
    print(f"Lineal      - Costo: {np.mean(lin_costs):.1f}, Tiempo: {np.mean(lin_times):.4f}s")
    
    plt.show()
