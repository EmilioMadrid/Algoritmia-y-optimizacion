import concurrent.futures
import time
import random
import matplotlib.pyplot as plt

def worker(samples):
    inside = 0
    for _ in range(samples):
        x = random.random()
        y = random.random()
        if x**2 + y**2 <= 1.0:
            inside += 1
    return inside

if __name__ == '__main__':
    processes_list = [1, 2, 4, 8]
    total_samples = 20_000_000 
    
    times = []
    for p in processes_list:
        samples_per_process = total_samples // p
        
        start_time = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=p) as executor:
            futures = [executor.submit(worker, samples_per_process) for _ in range(p)]
            results = [f.result() for f in futures]
        end_time = time.time()
        
        times.append(end_time - start_time)

    t1 = times[0]
    speedups = [t1 / t for t in times]

    plt.figure(figsize=(8, 6))
    plt.plot(processes_list, speedups, marker='o', linestyle='-', label='Speedup Real', color='blue', linewidth=2, markersize=8)
    plt.plot(processes_list, processes_list, linestyle='--', color='red', label='Speedup Ideal', linewidth=2)
    plt.xlabel('Número de Procesos', fontsize=12)
    plt.ylabel('Aceleración (Speedup)', fontsize=12)
    plt.title('Aceleración Paralela: Real vs. Ideal', fontsize=14)
    plt.xticks(processes_list)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    for p, t, s in zip(processes_list, times, speedups):
        print(f"Procesos: {p}, Tiempo: {t:.4f}s, Speedup: {s:.4f}x")
        
    plt.show()