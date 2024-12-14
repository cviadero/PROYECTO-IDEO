# -*- coding: utf-8 -*-
"""

@author: Carolina Viadero, Rodrigo Pagola, Eduardo Riveros, Ian Carbajal
PROYECTO FINAL DE INVESTIGACIÓN DE OPERACIONES
"""

import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


# Leer coordenadas desde el archivo, ignorando líneas con letras
def leer_coordenadas(archivo):
    coordenadas = []
    try:
        with open(archivo, 'r') as f:
            for linea in f:
                partes = linea.strip().split()
                if len(partes) >= 3:  
                    try:
                        coordenadas.append([float(partes[1]), float(partes[2])])
                    except ValueError:
                        continue 
        return np.array(coordenadas)
    except Exception as e:
        print(f"Error al leer el archivo {archivo}: {e}")
        return None


# Calcular matriz de distancias euclideanas redondeando al entero más cercano
def calcular_matriz_distancias(coordenadas):
    num_ciudades = len(coordenadas)
    matriz_distancias = np.full((num_ciudades, num_ciudades), np.inf)  # Inicializa con infinito
    for i in range(num_ciudades):
        for j in range(num_ciudades):
            if i != j:  # No calculamos distancias a sí mismas
                distancia = np.linalg.norm(coordenadas[i] - coordenadas[j])
                matriz_distancias[i, j] = int(round(distancia))  
    return matriz_distancias

#MÉTODO DEL ALGORITMO DE COLONIA DE HORMIGAS (ACO)

# Definimos la clase Colonia de Hormigas y sus atributos
class ColoniaDeHormigas:
    def __init__(self, distancias, num_hormigas, iteraciones, alfa, beta, evaporacion, Q):
        self.distancias = distancias
        self.feromonas = np.ones(self.distancias.shape) / len(distancias)
        self.num_hormigas = num_hormigas
        self.iteraciones = iteraciones
        self.alfa = alfa
        self.beta = beta
        self.evaporacion = evaporacion
        self.Q = Q
        self.num_ciudades = len(distancias)

    def _calcular_probabilidad(self, i, j, visitadas):
        if self.distancias[i][j] == np.inf or j in visitadas:
            return 0
        feromona = self.feromonas[i][j] ** self.alfa
        visibilidad = (1 / (self.distancias[i][j] + np.finfo(float).eps)) ** self.beta
        return feromona * visibilidad

    def _elegir_siguiente_ciudad(self, ciudad_actual, visitadas):
        probabilidades = np.zeros(self.num_ciudades)
        for j in range(self.num_ciudades):
            if j not in visitadas:  # Solo ciudades no visitadas
                probabilidades[j] = self._calcular_probabilidad(ciudad_actual, j, visitadas)
        suma_probabilidades = np.sum(probabilidades)
        if suma_probabilidades == 0:
            return None
        probabilidades /= suma_probabilidades
        return np.argmax(probabilidades)  # Selecciona la ciudad con la probabilidad más alta


     #Solución completa para una hormiga pasando una vez por nodo
    def _construir_solucion(self):
        solucion = [0]  # Fijamos siempre la ciudad inicial como la ciudad 1 con indice 0
        visitadas = set(solucion)
        while len(solucion) < self.num_ciudades:
            siguiente_ciudad = self._elegir_siguiente_ciudad(solucion[-1], visitadas)
            if siguiente_ciudad is None:
                break
            solucion.append(siguiente_ciudad)
            visitadas.add(siguiente_ciudad)
        return solucion if len(solucion) == self.num_ciudades else None


   
     #Paralelizamos la construcción de soluciones para todas las hormigas 
      #para que sea menor el tiempo de ejecución
    def _construir_soluciones(self):
        soluciones = Parallel(n_jobs=-1)(delayed(self._construir_solucion)() for _ in range(self.num_hormigas))
        return [s for s in soluciones if s is not None]  
         # Filtra solo soluciones válidas

    def _actualizar_feromonas(self, soluciones):
        self.feromonas *= (1 - self.evaporacion)
        for solucion in soluciones:
            costo = self._calcular_costo_solucion(solucion)
            for i in range(len(solucion) - 1):
                self.feromonas[solucion[i]][solucion[i + 1]] += self.Q / costo

    def _calcular_costo_solucion(self, solucion):
        return sum(self.distancias[solucion[i]][solucion[i + 1]] for i in range(len(solucion) - 1))

    def ejecutar(self):
        mejor_solucion = None
        mejor_costo = float('inf')
        iteraciones_sin_mejora = 0
        for _ in range(self.iteraciones):
            soluciones = self._construir_soluciones()
            for solucion in soluciones:
                costo = self._calcular_costo_solucion(solucion)
                if costo < mejor_costo:
                    mejor_solucion, mejor_costo = solucion, costo
                    iteraciones_sin_mejora = 0
                else:
                    iteraciones_sin_mejora += 1
            self._actualizar_feromonas(soluciones)
            if iteraciones_sin_mejora >= 10: 
                # Lo paramos antes si no vemos mejora para que sea menor el tiempo de ejecución
                break
        return mejor_solucion, mejor_costo


#MÉTODO DE LIN- KERNIGHAN

def lin_kernighan(distancias, ruta):
    n = len(ruta)
    mejora = True
    while mejora:
        mejora = False
        for i in range(n - 1):
            for j in range(i + 2, min(i + 10, n)):  # Limitar las aristas probadas
                delta = (distancias[ruta[i]][ruta[j]] + distancias[ruta[i + 1]][ruta[(j + 1) % n]]) - \
                        (distancias[ruta[i]][ruta[i + 1]] + distancias[ruta[j]][ruta[(j + 1) % n]])
                if delta < 0:
                    ruta[i + 1:j + 1] = reversed(ruta[i + 1:j + 1])
                    mejora = True
    return ruta


# Resolvemos el TSP combinando ambas heurísticas
def resolver_tsp(distancias):
    # Colonia de Hormigas
    colonia = ColoniaDeHormigas(distancias, num_hormigas=50, iteraciones=100, alfa=1, beta=5, evaporacion=0.5, Q=100)
    solucion_inicial, costo_inicial = colonia.ejecutar()
    print(f"Costo con ACO: {costo_inicial}")

    # Optimización con Lin-Kernighan
    solucion_optimizada = lin_kernighan(distancias, solucion_inicial)
    costo_optimizado = sum(distancias[solucion_optimizada[i]][solucion_optimizada[i + 1]] for i in range(len(solucion_optimizada) - 1))
    print(f"Costo optimizado con Lin-Kernighan: {costo_optimizado}")

    return solucion_optimizada, costo_optimizado


# Graficamos la ruta final obtenida tras Lin-Kernighan
def graficar_ruta(coordenadas, solucion):
    ruta = [coordenadas[ciudad] for ciudad in solucion + [solucion[0]]]
    ruta = np.array(ruta)
    
    plt.figure(figsize=(10, 6))
    plt.plot(ruta[:, 0], ruta[:, 1], 'b-', linewidth=2, label='Ruta')
    plt.scatter(ruta[:, 0], ruta[:, 1], c='red', s=50, label='Ciudades')
    
    for i, coord in enumerate(coordenadas):
        plt.text(coord[0], coord[1], str(i), fontsize=10, ha='right', va='bottom')
    
    plt.scatter(ruta[0, 0], ruta[0, 1], c='green', s=100, label='Inicio/Fin')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.title('Ruta Óptima Obtenida')
    plt.legend()
    plt.grid(True)
    plt.show()


# Ejecutamos los 3 archivos de texto con los datos
archivos = ["Qatar.txt", "Zimbabwe.txt", "Uruguay.txt"]
for archivo in archivos:
    coordenadas = leer_coordenadas(archivo)
    if coordenadas is not None:
        distancias = calcular_matriz_distancias(coordenadas)
        print(f"\nResolviendo TSP para {archivo}:")
        solucion, costo = resolver_tsp(distancias)
        print(f"\nRuta {archivo}: {solucion}\nCosto: {costo}")
        graficar_ruta(coordenadas, solucion)
