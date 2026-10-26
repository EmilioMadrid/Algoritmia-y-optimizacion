import matplotlib.pyplot as plt

def greedy_knapsack(items, capacity, sort_key, reverse_order):
    # Ordenar los objetos según el criterio especificado
    sorted_items = sorted(items, key=sort_key, reverse=reverse_order)
    
    total_value = 0
    total_weight = 0
    selected_items = []
    
    for item in sorted_items:
        if total_weight + item['weight'] <= capacity:
            selected_items.append(item['name'])
            total_weight += item['weight']
            total_value += item['value']
            
    return selected_items, total_weight, total_value

if __name__ == '__main__':
    # Definición de la instancia
    items = [
        {'name': 'A', 'weight': 2, 'value': 2},
        {'name': 'B', 'weight': 3, 'value': 3},
        {'name': 'C', 'weight': 10, 'value': 20}
    ]
    capacity = 10

    # Método 1: Menor peso primero (Ascendente)
    items_peso, peso_usado1, valor_total1 = greedy_knapsack(
        items, capacity, sort_key=lambda x: x['weight'], reverse_order=False
    )

    # Método 2: Mayor valor/peso primero (Descendente)
    items_ratio, peso_usado2, valor_total2 = greedy_knapsack(
        items, capacity, sort_key=lambda x: x['value'] / x['weight'], reverse_order=True
    )

    # Imprimir resultados en consola
    print(f"--- Criterio: Menor Peso ---")
    print(f"Objetos: {items_peso}, Capacidad Usada: {peso_usado1}kg, Valor Total: ${valor_total1}")
    print(f"\n--- Criterio: Mayor Valor/Peso ---")
    print(f"Objetos: {items_ratio}, Capacidad Usada: {peso_usado2}kg, Valor Total: ${valor_total2}")

    # Generar gráfica de barras
    etiquetas = ['Menor Peso Primero', 'Mayor Valor/Peso']
    valores = [valor_total1, valor_total2]
    pesos = [peso_usado1, peso_usado2]

    x = [0, 1]
    ancho = 0.35

    plt.figure(figsize=(8, 6))
    plt.bar([p - ancho/2 for p in x], valores, ancho, label='Valor Total ($)', color='green')
    plt.bar([p + ancho/2 for p in x], pesos, ancho, label='Capacidad Usada (kg)', color='orange')

    plt.ylabel('Cantidad')
    plt.title('Comparación de Estrategias Voraces (Mochila de 10kg)')
    plt.xticks(x, etiquetas)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.show()