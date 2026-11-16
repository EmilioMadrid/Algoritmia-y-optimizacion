import numpy as np
import random
import matplotlib.pyplot as plt

np.random.seed(42)
random.seed(42)

num_cities = 50
cities = np.random.rand(num_cities, 2) * 100

def distance_matrix(cities):
    diff = cities[:, np.newaxis, :] - cities[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))

dist_matrix = distance_matrix(cities)

def total_distance(tour, dist_matrix):
    return sum(dist_matrix[tour[i], tour[(i+1)%len(tour)]] for i in range(len(tour)))

def nearest_neighbor(dist_matrix, start_node):
    n = len(dist_matrix)
    unvisited = set(range(n))
    unvisited.remove(start_node)
    tour = [start_node]
    curr = start_node
    while unvisited:
        next_node = min(unvisited, key=lambda node: dist_matrix[curr, node])
        unvisited.remove(next_node)
        tour.append(next_node)
        curr = next_node
    return tour

def two_opt(initial_tour, dist_matrix):
    tour = list(initial_tour)
    n = len(tour)
    improvements = 0
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
                    improvements += 1
                    improved = True
                    break
            if improved:
                break
                
    return tour, improvements, total_distance(tour, dist_matrix)

if __name__ == '__main__':
    trials = 30
    random_improvements, random_qualities = [], []
    nn_improvements, nn_qualities = [], []

    for _ in range(trials):
        # 1. Inicio Aleatorio
        rand_tour = list(range(num_cities))
        random.shuffle(rand_tour)
        _, imp_r, qual_r = two_opt(rand_tour, dist_matrix)
        random_improvements.append(imp_r)
        random_qualities.append(qual_r)
        
        # 2. Inicio Vecino Más Cercano
        start_node = random.randint(0, num_cities - 1)
        nn_tour = nearest_neighbor(dist_matrix, start_node)
        _, imp_nn, qual_nn = two_opt(nn_tour, dist_matrix)
        nn_improvements.append(imp_nn)
        nn_qualities.append(qual_nn)

    # Generación de gráficas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.boxplot([random_improvements, nn_improvements], tick_labels=['Inicio Aleatorio', 'Inicio Vecino Cercano'])
    ax1.set_title('Número de Mejoras para Alcanzar Óptimo Local')
    ax1.set_ylabel('Cantidad de Mejoras (Iteraciones)')
    ax1.grid(True, linestyle=':', alpha=0.7)

    ax2.boxplot([random_qualities, nn_qualities], tick_labels=['Inicio Aleatorio', 'Inicio Vecino Cercano'])
    ax2.set_title('Calidad del Óptimo Local (Distancia del Tour)')
    ax2.set_ylabel('Distancia Total (menor es mejor)')
    ax2.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    
    print(f"Mejoras Promedio (Aleatorio): {np.mean(random_improvements):.1f}")
    print(f"Mejoras Promedio (Vecino Cercano): {np.mean(nn_improvements):.1f}")
    print(f"Distancia Final Promedio (Aleatorio): {np.mean(random_qualities):.1f}")
    print(f"Distancia Final Promedio (Vecino Cercano): {np.mean(nn_qualities):.1f}")
    
    plt.show()