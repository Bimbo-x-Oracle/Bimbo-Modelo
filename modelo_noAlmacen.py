import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime
import random
import os


input_dir = "original_bimbo_data"
output_dir = "output_data"

"""#**Pendientes**"""

def process_pending_data(file_path):
    # Cargar CSV
    df = pd.read_csv(file_path)

    ###scaler = MinMaxScaler()

    # Mantener ciertas columnas
    df = df[["FECHA DE CIERRE","CARGA", "ITEM", "CANTIDAD"]]

    # Crea un nuevo DataFrame 'dft' con solo las columnas "FECHA DE CIERRE" y "CARGA".
    dft = df[["FECHA DE CIERRE","CARGA"]]

    # Calcular el número de ocurrencias de cada valor único en la columna "CARGA".
    dfv = df["CARGA"].value_counts()

    # Agrupar los datos por "CARGA" y "ITEM", y sumar las cantidades de cada grupo.
    df = df.groupby(['CARGA', 'ITEM']).agg({'CANTIDAD': 'sum'}).reset_index()

    # Crear una tabla pivote donde las filas son "CARGA" y las columnas son "ITEM",
    # y los valores son la suma de "CANTIDAD". Los valores faltantes se completan con 0.
    dfl = df.pivot_table(index='CARGA', columns='ITEM', values='CANTIDAD', fill_value=0)
    dfl.columns = [col for col in dfl.columns]

    # Combinar el DataFrame de fechas y cargas únicas ('dft') con la tabla pivote 'dfl',
    dfl = pd.merge(dft.drop_duplicates(), dfl, on='CARGA', how="inner")

    # Combinar el conteo de cargas 'dfv' con el DataFrame resultante 'dfl',
    df = pd.merge(dfv, dfl, on='CARGA', how="inner")

    # Renombrar la columna 'count' a 'Pallet' para indicar el número de pallets por carga.
    df = df.rename(columns={'count': 'Pallet'})

    # Convertir la columna "FECHA DE CIERRE" a tipo datetime, usando el formato específico.
    df['FECHA DE CIERRE'] = pd.to_datetime(df['FECHA DE CIERRE'], format='%d/%m/%Y %H:%M:%S')

    # Restar la fecha actual a cada fecha de cierre, obteniendo una diferencia de tiempo.
    df['FECHA DE CIERRE'] = df['FECHA DE CIERRE'] - datetime.now()

    # Convertir la diferencia de tiempo en minutos y hacer que los valores sean positivos.
    df["FECHA DE CIERRE"] = df["FECHA DE CIERRE"].apply(lambda x: int(x.total_seconds() / 60))
    df["FECHA DE CIERRE"] = df["FECHA DE CIERRE"] * -1

    ###df["Pallet"] = scaler.fit_transform(df[['Pallet']])
    ###df["FECHA DE CIERRE"] = scaler.fit_transform(df[['FECHA DE CIERRE']])

    return df

pending = process_pending_data(f"{input_dir}/Pendientes.csv")

"""#**Órdenes**"""

import pandas as pd

def process_demand_data(file_path):
    # Cargar el archivo CSV
    df = pd.read_csv(file_path)

    # Filtrar las columnas necesarias
    df = df[["Orden", "Articulo", "Cantidad solicitada", "orderdtlstatus", "Precio de venta"]]

    # Filtrar las filas con 'orderdtlstatus' igual a 'Created'
    df = df[df['orderdtlstatus'] == 'Created']

    precio_total_por_orden = df.groupby('Orden')['Precio de venta'].sum().reset_index()

    # Agrupar los datos por 'Orden' y 'Articulo' para sumar las cantidades solicitadas
    df_agrupado = df.groupby(['Orden', 'Articulo'])['Cantidad solicitada'].sum().reset_index()

    # Pivotear el DataFrame para obtener una tabla con las órdenes como filas y los artículos como columnas
    df_ordenes = df_agrupado.pivot(index='Orden', columns='Articulo', values='Cantidad solicitada').fillna(0)

    df_ordenes = df_ordenes.reset_index()

    df_ordenes = pd.merge(df_ordenes, precio_total_por_orden, on='Orden', how='left')
    df_ordenes = df_ordenes.rename(columns={'Precio de venta': 'Precio de venta total'})


    return df_ordenes

# Ejecutar la función con el archivo proporcionado
df_ordenes = process_demand_data(f"{input_dir}/Ordenes.csv")

pending.to_csv(os.path.join(output_dir, 'pending_camiones_cleanformat.csv'), encoding='utf-8', index=False)
df_ordenes.to_csv(os.path.join(output_dir, 'ordenes_demanda_cleanformat.csv'), encoding='utf-8', index=False)

"""Modelo"""
def algoritmo_genetico_experiment(camiones_df, demanda_df, num_bahias=9, num_generaciones=300,
                                  tamaño_poblacion=150, prob_cruce=0.85, prob_mutacion=0.15, Tp=1):
    # Funciones internas
    def cargar_camiones(df_camiones):
        camiones = []
        for index, row in df_camiones.iterrows():
            camion = {
                'nombre': row['CARGA'],  # Asignar el ID del camión
                'Ai': row['FECHA DE CIERRE'],
                'Pi': row['Pallet'],
                'uij': row[3:].values.tolist(),
                'prioridad': 0
            }
            camiones.append(camion)
        return camiones

    def cargar_demanda(df_demanda):
        productos = []
        nombres_productos = df_demanda.columns[1:-1]
        for index, row in df_demanda.iterrows():
            producto = {
                'nombre': nombres_productos,
                'gamma': row['Precio de venta total'],
                'demanda_minima': row[1:-1].sum()
            }
            productos.append(producto)
        return productos

    def calcular_prioridad(camion, productos):
        return sum(producto['gamma'] * uij for uij, producto in zip(camion['uij'], productos))

    def inicializar_poblacion(num_camiones):
        poblacion = []
        for _ in range(tamaño_poblacion):
            camiones_disponibles = list(range(num_camiones))
            random.shuffle(camiones_disponibles)
            individuo = [[] for _ in range(num_bahias)]
            for bahia in range(num_bahias):
                num_camiones_bahia = len(camiones_disponibles) // (num_bahias - bahia)
                camiones_asignados = camiones_disponibles[:num_camiones_bahia]
                individuo[bahia] = camiones_asignados
                camiones_disponibles = camiones_disponibles[num_camiones_bahia:]
            poblacion.append(individuo)
        return poblacion

    def calcular_fitness(individuo, camiones, productos):
        fitness_total = 0
        tiempo_total = 0
        for bahia in individuo:
            tprevio = 0
            for camion_id in bahia:
                camion = camiones[camion_id]
                tiempo_servicio = camion['Pi'] * Tp
                ti = max(camion['Ai'], tprevio)
                fitness_total += camion['prioridad'] * (ti + tiempo_servicio)
                tiempo_total += (tiempo_servicio + ti - camion['Ai'])
                tprevio = ti + tiempo_servicio
        return fitness_total, tiempo_total

    def seleccion(poblacion, fitness):
        seleccionados = []
        for _ in range(tamaño_poblacion):
            torneo = random.sample(list(enumerate(fitness)), 3)
            ganador = max(torneo, key=lambda x: x[1][0])
            seleccionados.append(poblacion[ganador[0]])
        return seleccionados

    def cruce_parcialmente_mapeado(padre1, padre2):
        hijo1, hijo2 = [[] for _ in range(num_bahias)], [[] for _ in range(num_bahias)]
        usados_hijo1, usados_hijo2 = set(), set()
        for i in range(num_bahias):
            for camion in padre1[i]:
                if camion not in usados_hijo1:
                    hijo1[i].append(camion)
                    usados_hijo1.add(camion)
            for camion in padre2[i]:
                if camion not in usados_hijo2:
                    hijo2[i].append(camion)
                    usados_hijo2.add(camion)
        return hijo1, hijo2

    def mutacion_intercambio(individuo):
        bahias_no_vacias = [i for i, b in enumerate(individuo) if b]
        if len(bahias_no_vacias) > 1:
            bahia1, bahia2 = random.sample(bahias_no_vacias, 2)
            if individuo[bahia1] and individuo[bahia2]:
                camion1 = random.choice(individuo[bahia1])
                camion2 = random.choice(individuo[bahia2])
                individuo[bahia1].remove(camion1)
                individuo[bahia2].remove(camion2)
                individuo[bahia1].append(camion2)
                individuo[bahia2].append(camion1)
        return individuo

    def ejecutar_algoritmo(camiones, productos):
        for camion in camiones:
            camion['prioridad'] = calcular_prioridad(camion, productos)

        poblacion = inicializar_poblacion(len(camiones))
        mejor_individuo = None
        mejor_fitness = float('-inf')
        mejor_tiempo_total = None
        historico_fitness = []

        for generacion in range(num_generaciones):
            fitness = [calcular_fitness(individuo, camiones, productos) for individuo in poblacion]
            historico_fitness.extend([f[0] for f in fitness])
            poblacion_seleccionada = seleccion(poblacion, fitness)

            nueva_poblacion = []
            for i in range(0, len(poblacion_seleccionada) - 1, 2):
                padre1, padre2 = poblacion_seleccionada[i], poblacion_seleccionada[i + 1]

                # Aplicar probabilidad de cruce
                if random.random() < prob_cruce:
                    hijo1, hijo2 = cruce_parcialmente_mapeado(padre1, padre2)
                else:
                    hijo1, hijo2 = padre1[:], padre2[:]

                # Aplicar probabilidad de mutación
                nueva_poblacion.append(mutacion_intercambio(hijo1) if random.random() < prob_mutacion else hijo1)
                nueva_poblacion.append(mutacion_intercambio(hijo2) if random.random() < prob_mutacion else hijo2)

            # Si el tamaño de la población es impar, añade el último individuo sin modificar
            if len(poblacion_seleccionada) % 2 != 0:
                nueva_poblacion.append(poblacion_seleccionada[-1])

            poblacion = nueva_poblacion

            # Almacenar el mejor individuo de esta generación
            for ind, (fit, tiempo_total) in zip(poblacion, fitness):
                if fit > mejor_fitness:
                    mejor_fitness = fit
                    mejor_individuo = ind
                    mejor_tiempo_total = tiempo_total

        return mejor_individuo, mejor_tiempo_total

    # Cargar y preparar datos
    camiones = cargar_camiones(camiones_df)
    productos = cargar_demanda(demanda_df)

    # Ejecutar el algoritmo
    mejor_individuo, tiempo_total_ganador = ejecutar_algoritmo(camiones, productos)

    # En lugar de imprimir, devuelve los resultados
    return mejor_individuo, tiempo_total_ganador

# Llamada al algoritmo y almacenamiento de resultados
mejor_individuo, tiempo_total_ganador = algoritmo_genetico_experiment(pending.copy(), df_ordenes.copy())

# Uso posterior de los resultados en otras partes de la aplicación
print("Tiempo total final:", tiempo_total_ganador)
print("Orden de los camiones en las 9 bahías:", mejor_individuo)
