import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime
import random


"""#**Almacen**"""

def process_warehouse_data(file_path):
    # Cargar CSV
    warehouse = pd.read_csv(file_path, encoding='latin1')

    # Primera fila como títulos
    warehouse.columns = warehouse.iloc[0]
    warehouse = warehouse.drop(0).reset_index(drop=True)

    # Mantener ciertas columnas
    warehouse = warehouse[["PRODUCTO", "DISPONIBLE", "RECIBIDO", "UBICADO"]]

    # Cambiar columnas a numéricas y rellanar con 0
    warehouse['DISPONIBLE'] = pd.to_numeric(warehouse['DISPONIBLE'], errors='coerce').fillna(0)
    warehouse['UBICADO'] = pd.to_numeric(warehouse['UBICADO'], errors='coerce').fillna(0)
    warehouse['RECIBIDO'] = pd.to_numeric(warehouse['RECIBIDO'], errors='coerce').fillna(0)

    # Calcular RESERVA
    warehouse['RESERVA'] = warehouse['UBICADO'] + warehouse['RECIBIDO']

    # Mantener ciertas columnas
    warehouse = warehouse[["PRODUCTO", "DISPONIBLE", "RESERVA"]]

    return warehouse

warehouse = process_warehouse_data("Existencia.csv")
warehouse

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

pending = process_pending_data("Pendientes.csv")
pending

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
file_path = 'Ordenes.csv'  # Asegúrate de tener la ruta correcta al archivo
df_ordenes = process_demand_data(file_path)
print(df_ordenes)

df_ordenes = process_demand_data('Ordenes.csv')
df_ordenes

from google.colab import files

warehouse.to_csv('warehouse.csv', encoding='utf-8', index=False)
pending.to_csv('pending.csv', encoding='utf-8', index=False)
df_ordenes.to_csv('df_ordenes.csv', encoding='utf-8', index=False)
files.download('warehouse.csv')
files.download('pending.csv')
files.download('df_ordenes.csv')

"""#**Vaciar almacen**

#**Búsqueda de Cruce y Mutación**
"""

def algoritmo_genetico_experiment(camiones_df, demanda_df, num_bahias=9, num_generaciones=300,
                                  tamaño_poblacion=50, prob_cruce=0.9, prob_mutacion=0.1, Tp=2):
    # Funciones internas
    def cargar_camiones(df_camiones):
        camiones = []
        for index, row in df_camiones.iterrows():
            camion = {
                'nombre': row['CARGA'],
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

    def cruce_de_orden(padre1, padre2):
        hijo1 = [None] * len(padre1)
        hijo2 = [None] * len(padre2)
        inicio, fin = sorted(random.sample(range(len(padre1)), 2))
        hijo1[inicio:fin] = padre1[inicio:fin]
        hijo2[inicio:fin] = padre2[inicio:fin]
        def completar_hijo(hijo, otro_padre, inicio, fin):
            pos = fin
            for camion in otro_padre:
                if camion not in hijo:
                    if pos >= len(hijo):
                        pos = 0
                    hijo[pos] = camion
                    pos += 1
        completar_hijo(hijo1, padre2, inicio, fin)
        completar_hijo(hijo2, padre1, inicio, fin)
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

    def mutacion_inversion(individuo):
        bahia = random.choice([b for b in individuo if len(b) > 1])
        if len(bahia) > 1:
            i, j = sorted(random.sample(range(len(bahia)), 2))
            bahia[i:j+1] = reversed(bahia[i:j+1])
        return individuo

    def ejecutar_algoritmo(camiones, productos, cruce, mutacion):
        for camion in camiones:
            camion['prioridad'] = calcular_prioridad(camion, productos)

        poblacion = inicializar_poblacion(len(camiones))
        historico_fitness = []

        for generacion in range(num_generaciones):
            fitness = [calcular_fitness(individuo, camiones, productos) for individuo in poblacion]
            historico_fitness.extend([f[0] for f in fitness])

            poblacion_seleccionada = seleccion(poblacion, fitness)

            nueva_poblacion = []
            for i in range(0, tamaño_poblacion, 2):
                padre1, padre2 = poblacion_seleccionada[i], poblacion_seleccionada[i + 1]

                # Aplicar probabilidad de cruce
                if random.random() < prob_cruce:
                    hijo1, hijo2 = cruce(padre1, padre2)
                else:
                    hijo1, hijo2 = padre1[:], padre2[:]

                # Aplicar probabilidad de mutación
                nueva_poblacion.append(mutacion(hijo1) if random.random() < prob_mutacion else hijo1)
                nueva_poblacion.append(mutacion(hijo2) if random.random() < prob_mutacion else hijo2)

            poblacion = nueva_poblacion

        return max(historico_fitness)

    combinaciones = [
        ("Cruce Parcialmente Mapeado + Mutación Intercambio", cruce_parcialmente_mapeado, mutacion_intercambio),
        ("Cruce Parcialmente Mapeado + Mutación Inversión", cruce_parcialmente_mapeado, mutacion_inversion),
        ("Cruce de Orden + Mutación Intercambio", cruce_de_orden, mutacion_intercambio),
        ("Cruce de Orden + Mutación Inversión", cruce_de_orden, mutacion_inversion)
    ]

    resultados = {}
    scaler = MinMaxScaler()
    for name, cruce, mutacion in combinaciones:
        fitness_ganadores = []

        for semilla in range(30):
            random.seed(semilla)
            camiones = cargar_camiones(camiones_df)
            productos = cargar_demanda(demanda_df)
            fitness_ganador = ejecutar_algoritmo(camiones, productos, cruce, mutacion)
            fitness_ganadores.append(fitness_ganador)

        fitness_ganadores_escalados = scaler.fit_transform(np.array(fitness_ganadores).reshape(-1, 1)).flatten()
        promedio_fitness = np.mean(fitness_ganadores_escalados)
        desviacion_estandar_fitness = np.std(fitness_ganadores_escalados)

        resultados[name] = {
            'Promedio Fitness': promedio_fitness,
            'Desviación Estándar Fitness': desviacion_estandar_fitness
        }

    return resultados

def ejecutar_experimentos_multiples(camiones_data, demanda_data, num_problemas=6):
    descripciones = [
        "Configuración básica",
        "Aumento en la cantidad de camiones",
        "Reducción en la demanda total",
        "Incremento en la demanda de ciertos productos",
        "Número reducido de camiones",
        "Alta prioridad para algunos camiones y reducción en la cantidad de camiones"
    ]

    resultados_globales = []

    for problema in range(num_problemas):
        camiones_df = camiones_data.copy()
        demanda_df = demanda_data.copy()

        # Aplicar variaciones específicas a cada problema
        if problema == 1:
            camiones_df = pd.concat([camiones_df, camiones_df.sample(frac=0.2, random_state=42)], ignore_index=True)

        elif problema == 2:
            demanda_df.iloc[:, 1:-1] *= 0.7

        elif problema == 3:
            demanda_df.iloc[:, 1:6] *= 1.5

        elif problema == 4:
            camiones_df = camiones_df.sample(frac=0.5, random_state=42).reset_index(drop=True)

        elif problema == 5:
            camiones_df = camiones_df.sample(frac=0.5, random_state=42).reset_index(drop=True)
            camiones_df['Prioridad'] = 0
            camiones_df.loc[:4, 'Prioridad'] = 100

        resultados = algoritmo_genetico_experiment(camiones_df, demanda_df)

        resultado_problema = {
            #'Problema': f'Problema {problema + 1}',
            'Descripción': descripciones[problema]
        }

        for combinacion, valores in resultados.items():
            resultado_problema[f'{combinacion} - Promedio Fitness'] = valores['Promedio Fitness']
            resultado_problema[f'{combinacion} - Desviación Estándar Fitness'] = valores['Desviación Estándar Fitness']

        resultados_globales.append(resultado_problema)

    resultados_df = pd.DataFrame(resultados_globales)
    return resultados_df

# Cargar datasets
pending = pd.read_csv('pending.csv')
orders = pd.read_csv('df_ordenes.csv')

# Ejecutar los experimentos y mostrar el DataFrame con descripciones y estilo
resultados_df = ejecutar_experimentos_multiples(pending.copy(), orders.copy())

# Aplicar estilo para una visualización mejorada
styled_resultados_df = (resultados_df.style
                        .set_properties(**{'text-align': 'center', 'font-size': '12pt'})
                        .set_caption("Resultados de Experimentos Genéticos por Problema y Combinación con Descripción"))

# Mostrar el DataFrame
styled_resultados_df

resultados_df.to_csv('resultados_df.csv', index=False)

def graficar_fitness_por_problema(df):

    # Definir las combinaciones y colores
    combinaciones = [
        "Cruce Parcialmente Mapeado + Mutación Intercambio",
        "Cruce Parcialmente Mapeado + Mutación Inversión",
        "Cruce de Orden + Mutación Intercambio",
        "Cruce de Orden + Mutación Inversión"
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Obtener los problemas desde la columna 'Descripción'
    problemas = df["Descripción"].values
    num_problemas = len(problemas)
    num_combinaciones = len(combinaciones)

    # Crear listas de listas para almacenar datos de fitness por combinación y problema
    fitness_data = [[] for _ in range(num_combinaciones)]
    for idx, combinacion in enumerate(combinaciones):
        promedio_col = f"{combinacion} - Promedio Fitness"
        desviacion_col = f"{combinacion} - Desviación Estándar Fitness"

        for i in range(num_problemas):
            promedio = df.loc[i, promedio_col]
            desviacion = df.loc[i, desviacion_col]
            # Simular datos de fitness alrededor del promedio usando la desviación estándar
            valores_fitness = np.random.normal(loc=promedio, scale=desviacion, size=10)
            fitness_data[idx].append(valores_fitness)

    # Crear el gráfico de boxplot
    plt.figure(figsize=(20, 12))

    # Generar boxplots para cada combinación de cada problema
    for idx, (data, color) in enumerate(zip(fitness_data, colors)):
        for i, valores in enumerate(data):
            plt.boxplot(
                valores,
                positions=[i * (num_combinaciones + 1) + idx],
                widths=0.6,
                patch_artist=True,
                boxprops=dict(facecolor=color),
                medianprops=dict(color="black")
            )

    # Configuración de etiquetas y títulos
    plt.xticks(np.arange(num_problemas) * (num_combinaciones + 1) + 1.5, problemas, rotation=45)
    plt.xlabel("Problema")
    plt.ylabel("Fitness")
    plt.title("Distribución de Fitness por Problema y Combinación de Cruce/Mutación")

    # Leyenda con colores y combinaciones
    for idx, (combinacion, color) in enumerate(zip(combinaciones, colors)):
        plt.plot([], [], color=color, label=combinacion, linewidth=10)
    plt.legend(loc="upper right", bbox_to_anchor=(1, 1))

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

# Ejecutar la función con el DataFrame cargado
graficar_fitness_por_problema(resultados_df)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

def graficar_bayesian_signed_rank(df):
    # Definir las combinaciones a evaluar
    combinaciones = [
        "Cruce Parcialmente Mapeado + Mutación Intercambio",
        "Cruce Parcialmente Mapeado + Mutación Inversión",
        "Cruce de Orden + Mutación Intercambio",
        "Cruce de Orden + Mutación Inversión"
    ]

    # Obtener los problemas desde la columna 'Descripción'
    problemas = df["Descripción"].values
    num_problemas = len(problemas)
    num_combinaciones = len(combinaciones)

    # Almacenar los resultados de diferencia de fitness entre combinaciones
    diferencias_fitness = np.zeros((num_problemas, num_combinaciones))

    # Iterar sobre cada problema y calcular las diferencias bayesianas
    for i in range(num_problemas):
        for idx, combinacion in enumerate(combinaciones):
            promedio_col = f"{combinacion} - Promedio Fitness"
            promedio_fitness = df.loc[i, promedio_col]
            # Suponiendo una varianza constante, podemos simplificar
            # Generamos valores aleatorios como diferencias observadas en el fitness
            diferencia = promedio_fitness - np.mean(df[promedio_col])
            diferencias_fitness[i, idx] = diferencia

    # Calcular intervalos de credibilidad (95%) para cada combinación de problemas
    intervalos_credibilidad = norm.interval(0.95, loc=diferencias_fitness.mean(axis=0), scale=diferencias_fitness.std(axis=0))

    # Crear el gráfico
    plt.figure(figsize=(12, 8))
    for idx, (combinacion, (lower, upper)) in enumerate(zip(combinaciones, zip(*intervalos_credibilidad))):
        plt.errorbar(
            x=problemas,
            y=diferencias_fitness.mean(axis=0),
            yerr=[diferencias_fitness.mean(axis=0) - lower, upper - diferencias_fitness.mean(axis=0)],
            label=combinacion,
            fmt='o',
            capsize=5
        )

    # Etiquetas y título
    plt.xlabel("Problema")
    plt.ylabel("Diferencia Promedio de Fitness")
    plt.title("Distribución de Fitness - Prueba Bayesian Signed Rank")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()

# Ejecutar la función con el DataFrame cargado
graficar_bayesian_signed_rank(resultados_df)

"""#**Búsqueda de probabilidades para cruce y mutación, y tamaño de población**"""

def algoritmo_genetico_experiment(camiones_df, demanda_df, num_bahias=9, num_generaciones=300,
                                  tamaño_poblacion=50, Tp=2):
    # Funciones internas
    def cargar_camiones(df_camiones):
        camiones = []
        for index, row in df_camiones.iterrows():
            camion = {
                'nombre': row['CARGA'],
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


    def ejecutar_algoritmo(camiones, productos, prob_cruce, prob_mutacion):
        for camion in camiones:
            camion['prioridad'] = calcular_prioridad(camion, productos)

        poblacion = inicializar_poblacion(len(camiones))
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

            if len(poblacion_seleccionada) % 2 != 0:
                nueva_poblacion.append(poblacion_seleccionada[-1])

            poblacion = nueva_poblacion

        return max(historico_fitness)

    # Definir las combinaciones de probabilidades de cruce y mutación
    combinaciones_probabilidades = [(0.85, 0.15), (0.9, 0.1), (0.95, 0.05)]
    resultados_globales = {}

    # Probar diferentes tamaños de población
    for tamaño_poblacion in [50, 75, 100, 125, 150]:
        resultados_fila = {}
        for prob_cruce, prob_mutacion in combinaciones_probabilidades:
            fitness_ganadores = []
            for semilla in range(30):
                random.seed(semilla)
                camiones = cargar_camiones(camiones_df)
                productos = cargar_demanda(demanda_df)
                fitness_ganador = ejecutar_algoritmo(camiones, productos, prob_cruce, prob_mutacion)
                fitness_ganadores.append(fitness_ganador)

            # Escalar los resultados
            scaler = MinMaxScaler()
            fitness_ganadores_escalados = scaler.fit_transform(np.array(fitness_ganadores).reshape(-1, 1)).flatten()
            promedio_fitness = np.mean(fitness_ganadores_escalados)
            resultados_fila[f'P({prob_cruce}, M{prob_mutacion})'] = promedio_fitness

        resultados_globales[f'Tamaño Población {tamaño_poblacion}'] = resultados_fila

    # Crear un DataFrame para los resultados
    resultados_df = pd.DataFrame(resultados_globales).T
    return resultados_df

# Cargar datasets
pending = pd.read_csv('pending.csv')
orders = pd.read_csv('orders.csv')

# Ejecutar los experimentos y mostrar el DataFrame
resultados_df2 = algoritmo_genetico_experiment(pending.copy(), orders.copy())
resultados_df2.style.set_properties(**{'text-align': 'center', 'font-size': '12pt'}).set_caption("Resultados de Experimentos Genéticos por Tamaño de Población y Combinación de Probabilidades")

resultados_df2

# Crear el gráfico de barras
resultados_df2.plot(kind='bar', figsize=(10, 6))

# Personalizar el gráfico
plt.title('Resultados de Experimentos Genéticos por Tamaño de Población y Combinación de Probabilidades')
plt.xlabel('Tamaño de Población')
plt.ylabel('Promedio de Fitness (Escalado)')
plt.xticks(rotation=45)
plt.legend(title='Combinaciones de P(Cruce, Mutación)')
plt.tight_layout()

# Mostrar el gráfico
plt.show()

"""#**Algoritmo genético**"""

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

    # Mostrar resultados
    print(f"Tiempo total final: {tiempo_total_ganador}")
    print("\nOrden de los camiones en las 9 bahías:")
    for i, bahia in enumerate(mejor_individuo, start=1):
        bahia_ids = [camiones[camion_id]['nombre'] for camion_id in bahia]
        print(f"Bahía {i}: {bahia_ids}")

# Cargar datasets
pending = pd.read_csv('pending.csv')
orders = pd.read_csv('orders.csv')

# Ejecutar el algoritmo genético y mostrar el orden de los camiones y el tiempo total
algoritmo_genetico_experiment(pending.copy(), orders.copy())