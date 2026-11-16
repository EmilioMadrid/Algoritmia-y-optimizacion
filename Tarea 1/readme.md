# Prácticas de Algoritmia y Optimización
Repositorio de ejercicios prácticos que abordan computación paralela, algoritmos voraces, búsqueda local, recocido simulado y algoritmos de aproximación en Python.

---

## 🛠️ Requisitos y Librerías

Para ejecutar todos los scripts de este repositorio, asegúrate de tener instalado **Python 3.8 o superior**. 

Las bibliotecas necesarias son `numpy` y `matplotlib`. Puedes instalarlas ejecutando el siguiente comando en tu terminal:

```bash
pip install numpy matplotlib
```

---

## 🚀 Instrucciones de Ejecución

Clona o descarga este repositorio en tu computadora, abre una terminal en la carpeta contenedora y sigue las instrucciones para cada ejercicio:

### 1. Ejercicio de Aceleración Paralela (Speedup)
Simula el cálculo de Pi por Monte Carlo evaluando el rendimiento con 1, 2, 4 y 8 procesos.
* **Comando:**
  ```bash
  python run_pi.py
  ```
* **Salida esperada:** Imprime los tiempos y *speedups* en consola y genera la gráfica comparativa.

---

### 2. Ejercicio de Análisis Algorítmico (Knapsack Voraz)
Compara la heurística de "Menor Peso" frente a "Mayor Valor/Peso" en el problema de la mochila.
* **Comando:**
  ```bash
  python mochila.py
  ```
* **Salida esperada:** Muestra los resultados en consola y abre una ventana con la gráfica de barras comparativa.

---

### 3. Ejercicio de Búsqueda Local (2-opt para TSP)
Evalúa el rendimiento de la heurística 2-opt partiendo de soluciones iniciales aleatorias frente al Vecino Más Cercano.
* **Comando:**
  ```bash
  python tsp_2opt.py
  ```
* **Salida esperada:** Imprime los promedios de iteraciones y distancias en consola, generando diagramas de caja duales.

---

### 4. Ejercicio de Ajuste de Parámetros (Recocido Simulado)
Implementa *Simulated Annealing* comparando los esquemas de enfriamiento exponencial y lineal.
* **Comando:**
  ```bash
  python simulated_annealing.py
  ```
* **Salida esperada:** Ejecuta múltiples pruebas estadísticas y genera las curvas de convergencia junto con el diagrama de dispersión.

---

### 5. Ejercicio de Razón de Aproximación (MST Approximation)
Evalúa el algoritmo de aproximación basado en Árboles de Expansión Mínima frente al óptimo exacto en múltiples instancias.
* **Comando:**
  ```bash
  python mst_approx.py
  ```
* **Salida esperada:** Calcula las razones de aproximación, imprime las métricas promedio y grafica el histograma y los diagramas de caja comparativos.