import random
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CONFIGURACIÓN INICIAL Y GENERACIÓN DE R
# ==========================================
MATRICULA = 2086253
random.seed(MATRICULA)
np.random.seed(MATRICULA)

# Generar un único número aleatorio r ~ Uniforme(0, 1) y fijarlo
r = random.random()
print(f"Número aleatorio r fijo (semilla {MATRICULA}): {r:.4f}")

# Definir rango de temperatura del Recocido Simulado
T_min, T_max = 10.0, 500.0

# Calcular fracciones para Opción A (3 niveles)
frac_bajo = 0.3 * r
frac_medio = 0.3 * r + 0.3
frac_alto = 0.3 * r + 0.6

# Convertir fracciones a temperaturas reales
niveles_temp = {
    'Bajo': T_min + frac_bajo * (T_max - T_min),
    'Medio': T_min + frac_medio * (T_max - T_min),
    'Alto': T_min + frac_alto * (T_max - T_min)
}

print("\nNiveles de temperatura calculados:")
for nivel, temp in niveles_temp.items():
    print(f"  - {nivel}: {temp:.2f}")

# ==========================================
# 2. ALGORITMO TSP, 2-OPT Y RECOCIDO SIMULADO
# ==========================================
def crear_instancia_tsp(n, seed):
    np.random.seed(seed)
    return np.random.rand(n, 2) * 100

def calcular_costo_ruta(ruta, coords):
    costo = 0.0
    n = len(ruta)
    for i in range(n):
        p1 = coords[ruta[i]]
        p2 = coords[ruta[(i + 1) % n]]
        costo += np.linalg.norm(p1 - p2)
    return costo

def vecino_2opt(ruta):
    n = len(ruta)
    i, j = sorted(random.sample(range(n), 2))
    nueva_ruta = ruta.copy()
    nueva_ruta[i:j+1] = reversed(nueva_ruta[i:j+1])
    return nueva_ruta

def recocido_simulado(coords, T0, iter_max=1000, alpha=0.98):
    n = len(coords)
    actual = list(range(n))
    random.shuffle(actual)
    costo_actual = calcular_costo_ruta(actual, coords)
    
    mejor = actual.copy()
    costo_mejor = costo_actual
    
    T = T0
    for _ in range(iter_max):
        siguiente = vecino_2opt(actual)
        costo_sig = calcular_costo_ruta(siguiente, coords)
        delta = costo_sig - costo_actual
        
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-8)):
            actual = siguiente
            costo_actual = costo_sig
            if costo_actual < costo_mejor:
                mejor = actual.copy()
                costo_mejor = costo_actual
                
        T *= alpha
    return costo_mejor

# ==========================================
# 3. EJECUCIÓN DEL DISEÑO EXPERIMENTAL (90 CORRIDAS)
# ==========================================
tamanios = [5, 10, 15]
replicas = 10
datos = []

for n in tamanios:
    # Generar una instancia fija por tamaño usando una semilla base
    coords_instancia = crear_instancia_tsp(n, seed=42 + n)
    
    for temp_nombre, temp_valor in niveles_temp.items():
        for rep in range(replicas):
            # Semilla distinta por réplica para capturar aleatoriedad
            seed_replica = 1000 * n + 100 * rep + int(temp_valor)
            random.seed(seed_replica)
            np.random.seed(seed_replica)
            
            costo = recocido_simulado(coords_instancia, T0=temp_valor)
            
            datos.append({
                'tamano': str(n),
                'temperatura': temp_nombre,
                'temp_valor': temp_valor,
                'replica': rep,
                'costo': costo
            })

df_experimento = pd.DataFrame(datos)
print(f"\nTotal de corridas ejecutadas: {len(df_experimento)}")

# ==========================================
# 4. ANOVA DE DOS FACTORES
# ==========================================
modelo = ols('costo ~ C(temperatura) * C(tamano)', data=df_experimento).fit()
tabla_anova = sm.stats.anova_lm(modelo, typ=2)
print("\n--- TABLA ANOVA DE DOS FACTORES ---")
print(tabla_anova)

# ==========================================
# 5. CÁLCULO DE MEDIANA BOOTSTRAP E IC (95%)
# ==========================================
def bootstrap_mediana(data, num_boot=1000):
    boot_medianas = []
    n_len = len(data)
    for _ in range(num_boot):
        muestra = np.random.choice(data, size=n_len, replace=True)
        boot_medianas.append(np.median(muestra))
    ic_inf = np.percentile(boot_medianas, 2.5)
    ic_sup = np.percentile(boot_medianas, 97.5)
    return np.median(data), ic_inf, ic_sup

resultados_resumen = []
for n in tamanios:
    for temp_nombre in niveles_temp.keys():
        subset = df_experimento[(df_experimento['tamano'] == str(n)) & (df_experimento['temperatura'] == temp_nombre)]['costo'].values
        med, inf, sup = bootstrap_mediana(subset)
        resultados_resumen.append({
            'Tamano': n,
            'Temperatura': temp_nombre,
            'Mediana_Costo': med,
            'IC_Inferior': inf,
            'IC_Superior': sup
        })

df_resumen = pd.DataFrame(resultados_resumen)
print("\n--- RESUMEN DE MEDIANAS Y BOOTSTRAP ---")
print(df_resumen)

# ==========================================
# 6. VISUALIZACIÓN DE RESULTADOS (GRÁFICAS)
# ==========================================
plt.figure(figsize=(14, 5))

# Gráfica de Efectos Principales - Temperatura
plt.subplot(1, 2, 1)
sns.lineplot(data=df_experimento, x='temperatura', y='costo', hue='tamano', marker='o', ci=95)
plt.title('Efectos de Interacción: Temperatura vs Tamaño')
plt.xlabel('Nivel de Temperatura')
plt.ylabel('Costo de la Ruta')

# Gráfica de Interacción
plt.subplot(1, 2, 2)
sns.pointplot(data=df_experimento, x='temperatura', y='costo', hue='tamano', dodge=True, capsize=.1)
plt.title('Gráfica de Interacción (Temperatura × Tamaño)')
plt.xlabel('Temperatura')
plt.ylabel('Costo Promedio')

plt.tight_layout()
plt.savefig('graficas_doe_tsp.png', dpi=300)
plt.show()