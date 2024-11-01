import matplotlib.pyplot as plt
import numpy as np
import csv

# File CSV da cui leggere i dati
csv_file_parallelo = 'parallelo.csv'
csv_file_sequenziale = 'sequenziali.csv'

# Inizializzazione di x1, y1 e y2
x1 = []
y1 = []
y2 = []

# Lettura di res1.csv (valori CPU)
with open(csv_file_parallelo, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Salta l'intestazione
    for row in reader:
        x1.append(float(row[0]))  # Dimensione del problema
        y1.append(float(row[1]))  # Tempo di esecuzione CPU

# Lettura di res2.csv (valori GPU)
with open(csv_file_sequenziale, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Salta l'intestazione
    for row in reader:
        y2.append(float(row[1]))  # Tempo di esecuzione GPU

# Converti x1, y1 e y2 in array NumPy per plotting
x1 = np.array(x1)
y1 = np.array(y1)
y2 = np.array(y2)


# Creazione del grafico
plt.plot(x1, y1, label='GPU')  # label per la curva CPU
plt.plot(x1, y2, label='Optimized GPU')  # label per la curva CPU

# Segna il punto di intersezione
plt.xlabel('Problem Dimension')
plt.ylabel('Time Execution Measured (s)')
plt.title('CPU vs GPU Speed')
plt.legend()                    # mostra la legenda
plt.grid(True)                  # mostra una griglia

# Visualizzare il grafico
plt.show()