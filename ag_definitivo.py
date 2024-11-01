import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import random

# Definir directorios de input y output
input_dir = "datasets/original_bimbo_data"
output_dir = "datasets/output_data"

"""**Almacen**"""

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

warehouse = process_warehouse_data(f"{input_dir}/Existencia.csv")
warehouse

"""**Pendientes**"""

def process_pending_data(file_path):
    # Cargar CSV
    df = pd.read_csv(file_path)

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

    return df

pending = process_pending_data(f"{input_dir}/Pendientes.csv")
pending

"""**Órdenes**"""

def process_demand_data(file_path):
    # Cargar el archivo CSV
    df = pd.read_csv(file_path)

    scaler = MinMaxScaler()

    # Filtrar las columnas necesarias
    df = df[["Orden", "Articulo", "Texto breve personalizado de detalle 5",
             "Cantidad solicitada", "orderdtlstatus", "Precio de venta"]]

    # Filtrar las filas con 'orderdtlstatus' igual a 'Created'
    df = df[df['orderdtlstatus'] == 'Created']

    # Separar en dos DataFrames según el valor de 'Texto breve personalizado de detalle 5'
    df_pk = df[df['Texto breve personalizado de detalle 5'] == 'PK']
    df_fp = df[df['Texto breve personalizado de detalle 5'] == 'FP']

    # Calcular el precio total por orden para 'PK' y 'FP'
    precio_total_por_orden_pk = df_pk.groupby('Orden')['Precio de venta'].sum().reset_index()
    precio_total_por_orden_fp = df_fp.groupby('Orden')['Precio de venta'].sum().reset_index()

    # Agrupar y pivotear los datos de 'PK'
    df_pk = df_pk.groupby(['Orden', 'Articulo']).agg({'Cantidad solicitada': 'sum'}).reset_index()
    dfpk = df_pk.pivot(index='Orden', columns='Articulo', values='Cantidad solicitada').fillna(0)

    # Restablecer el índice para que 'Orden' vuelva a ser una columna regular
    dfpk = dfpk.reset_index()

    # Añadir el precio total por orden para 'PK'
    dfpk = pd.merge(dfpk, precio_total_por_orden_pk, on='Orden', how='left')
    dfpk = dfpk.rename(columns={'Precio de venta': 'Precio de venta total'})
    dfpk["Precio de venta total"] = scaler.fit_transform(dfpk[['Precio de venta total']])

    # Agrupar y pivotear los datos de 'FP'
    df_fp = df_fp.groupby(['Orden', 'Articulo']).agg({'Cantidad solicitada': 'sum'}).reset_index()
    dffp = df_fp.pivot(index='Orden', columns='Articulo', values='Cantidad solicitada').fillna(0)

    # Restablecer el índice para que 'Orden' vuelva a ser una columna regular
    dffp = dffp.reset_index()

    # Añadir el precio total por orden para 'FP'
    dffp = pd.merge(dffp, precio_total_por_orden_fp, on='Orden', how='left')
    dffp = dffp.rename(columns={'Precio de venta': 'Precio de venta total'})
    dffp["Precio de venta total"] = scaler.fit_transform(dffp[['Precio de venta total']])

    # Regresar los dos DataFrames
    return dfpk, dffp

dfpk, dffp = process_demand_data(f"{input_dir}/Ordenes.csv")
dfpk

warehouse.to_csv(f'{output_dir}/warehouse_existencias_cleanformat.csv', encoding='utf-8', index=False)
pending.to_csv(f'{output_dir}/pending_camiones_cleanformat.csv', encoding='utf-8', index=False)
dfpk.to_csv(f'{output_dir}/dfpk_ordenes_picking_cleanformat.csv', encoding='utf-8', index=False)
dffp.to_csv(f'{output_dir}/dffp_ordenes_fullpallet_cleanformat.csv', encoding='utf-8', index=False)

"""**Vaciar almacen**"""

# Función para asignar productos pk
def asignar_productos(dfpk, warehouse):
    # Iterar por cada orden en dfpk
    for idx, order in dfpk.iterrows():
        print(f"\nProcesando orden {order['Orden']}:")
        # Lista para registrar los productos suministrados en esta orden
        productos_suministrados = []
        # Iterar sobre cada producto en la orden actual
        for product in dfpk.columns[1:-1]:  # Evitar las columnas no relacionadas con productos
            demanda = order[product]
            # Si hay demanda en esta orden para este producto
            if demanda > 0:
                # Verificar si el producto está en el dataset de warehouse
                warehouse_row = warehouse[warehouse['PRODUCTO'] == int(product)]
                if not warehouse_row.empty:
                    stock_disponible = warehouse_row.iloc[0]['DISPONIBLE']
                    if stock_disponible >= demanda:
                        # Satisfacer la demanda completa
                        dfpk.loc[idx, product] = demanda
                        warehouse.loc[warehouse['PRODUCTO'] == int(product), 'DISPONIBLE'] -= demanda
                        productos_suministrados.append(f"Producto {product}: suministrado {demanda}")
                    else:
                        # Satisfacer lo que sea posible
                        dfpk.loc[idx, product] = stock_disponible
                        warehouse.loc[warehouse['PRODUCTO'] == int(product), 'DISPONIBLE'] = 0
                        productos_suministrados.append(f"Producto {product}: suministrado {stock_disponible}")

        # Imprimir el resultado para esta orden
        if productos_suministrados:
            print(f"Productos suministrados para la orden {order['Orden']}:")
            for producto in productos_suministrados:
                print(producto)
        else:
            print(f"No se suministraron productos para la orden {order['Orden']}.")

    return dfpk, warehouse

# Procesar con ordenes picking
# Carga los datasets (asegúrate de tener los archivos en la misma carpeta o de poner el path correcto)
warehouse = pd.read_csv(f'{output_dir}/warehouse_existencias_cleanformat.csv')
dfpk = pd.read_csv(f'{output_dir}/dfpk_ordenes_picking_cleanformat.csv')

# Ejecutar el algoritmo
dfpk_actualizado, warehouse_actualizado = asignar_productos(dfpk.copy(), warehouse.copy())

# Devolver los datasets actualizados
dfpk_actualizado.to_csv(f'{output_dir}/dfpk_ordenes_picking_cleanformat_resta_existencias.csv', index=False)
warehouse_actualizado.to_csv(f'{output_dir}/warehouse_existencias_cleanformat_dfpk_processed.csv', index=False)

warehouse_actualizado.head()

# Función para asignar productos fp
def asignar_productos(dffp, warehouse):
    # Iterar por cada orden en dffp
    for idx, order in dffp.iterrows():
        print(f"\nProcesando orden {order['Orden']}:")
        # Lista para registrar los productos suministrados en esta orden
        productos_suministrados = []
        # Iterar sobre cada producto en la orden actual
        for product in dffp.columns[1:-1]:  # Evitar las columnas no relacionadas con productos
            demanda = order[product]
            # Si hay demanda en esta orden para este producto
            if demanda > 0:
                # Verificar si el producto está en el dataset de warehouse
                warehouse_row = warehouse[warehouse['PRODUCTO'] == int(product)]
                if not warehouse_row.empty:
                    stock_disponible = warehouse_row.iloc[0]['RESERVA']  # Columna 3 (RESERVA)
                    if stock_disponible >= demanda:
                        # Satisfacer la demanda completa
                        dffp.loc[idx, product] = demanda
                        warehouse.loc[warehouse['PRODUCTO'] == int(product), 'RESERVA'] -= demanda
                        productos_suministrados.append(f"Producto {product}: suministrado {demanda}")
                    else:
                        # Satisfacer lo que sea posible
                        dffp.loc[idx, product] = stock_disponible
                        warehouse.loc[warehouse['PRODUCTO'] == int(product), 'RESERVA'] = 0
                        productos_suministrados.append(f"Producto {product}: suministrado {stock_disponible}")

        # Imprimir el resultado para esta orden
        if productos_suministrados:
            print(f"Productos suministrados para la orden {order['Orden']}:")
            for producto in productos_suministrados:
                print(producto)
        else:
            print(f"No se suministraron productos para la orden {order['Orden']}.")

    return dffp, warehouse

# Procesar con ordenes full pallet
# Carga los datasets (asegúrate de tener los archivos en la misma carpeta o de poner el path correcto)
warehouse = pd.read_csv(f'{output_dir}/warehouse_existencias_cleanformat_dfpk_processed.csv')
dffp = pd.read_csv(f'{output_dir}/dffp_ordenes_fullpallet_cleanformat.csv')

# Ejecutar el algoritmo
dffp_actualizado, warehouse_actualizado_2 = asignar_productos(dffp.copy(), warehouse.copy())

# Devolver los datasets actualizados
dffp_actualizado.to_csv(f'{output_dir}/dffp_ordenes_fullpallet_cleanformat_resta_existencias.csv', index=False)
warehouse_actualizado_2.to_csv(f'{output_dir}/warehouse_existencias_cleanformat_dfpk_dffp_processed.csv', index=False)

warehouse_actualizado_2.head()

def merge_and_sum_datasets(dffp, dfpk) -> pd.DataFrame:
    # Combinar ambos datasets
    merged_df = pd.merge(dffp, dfpk, on='Orden', how='outer', suffixes=('_dffp', '_dfpk'))

    # Sumas artículos
    for column in merged_df.columns:
        if '_dffp' in column:
            base_column = column.replace('_dffp', '')
            if f'{base_column}_dfpk' in merged_df.columns:
                merged_df[base_column] = merged_df[column].fillna(0) + merged_df[f'{base_column}_dfpk'].fillna(0)
                merged_df.drop([column, f'{base_column}_dfpk'], axis=1, inplace=True)

    merged_df = merged_df.fillna(0)

    return merged_df

dffp1 = pd.read_csv(f'{output_dir}/dffp_ordenes_fullpallet_cleanformat_resta_existencias.csv')
dfpk1 = pd.read_csv(f'{output_dir}/dfpk_ordenes_picking_cleanformat_resta_existencias.csv')

orders = merge_and_sum_datasets(dffp.copy(), dfpk.copy())

# Devolver los datasets actualizados
orders.to_csv(f'{output_dir}/orders_cleanformat_resta_existencias.csv', index=False) #demanda actualizada

orders.head()

"""**Algoritmo genético**"""

def algoritmo_genetico(camiones_df, demanda_df, num_bahias=9, num_generaciones=300,
                                tamaño_poblacion=50, prob_cruce=0.8, prob_mutacion=0.2, Tp=2):

    # Function to load trucks data
    def cargar_camiones(df_camiones):
        camiones = []
        for index, row in df_camiones.iterrows():
            camion = {
                'nombre': row['CARGA'],  # Name of the truck (assuming it's in the column 'CARGA')
                'Ai': row['FECHA DE CIERRE'],  # Time of arrival
                'Pi': row['Pallet'],  # Number of pallets
                'uij': row[3:].values.tolist(),  # Product quantities starting from the 3rd column
                'prioridad': 0  # Priority will be calculated later
            }
            camiones.append(camion)
        return camiones

    # Function to load product demand data
    def cargar_demanda(df_demanda):
        productos = []
        nombres_productos = df_demanda.columns[1:-1]  # Assuming product names are the column headers (excluding 'Orden' and 'Precio de venta total')
        for index, row in df_demanda.iterrows():
            producto = {
                'nombre': nombres_productos,  # Names of the products
                'gamma': row['Precio de venta total'],  # Set 'gamma' as 'Precio de venta total'
                'demanda_minima': row[1:-1].sum()  # Exclude 'Orden' and 'Precio de venta total'
            }
            productos.append(producto)
        return productos

    # Load truck and demand data
    camiones = cargar_camiones(camiones_df)
    productos = cargar_demanda(demanda_df)

    # Function to calculate priority of each truck
    def calcular_prioridad(camion, productos):
        return sum(producto['gamma'] * uij for uij, producto in zip(camion['uij'], productos))

    # Function to initialize the population without repeating trucks
    def inicializar_poblacion(num_camiones):
        poblacion = []
        for _ in range(tamaño_poblacion):
            camiones_disponibles = list(range(num_camiones))
            random.shuffle(camiones_disponibles)
            individuo = [[] for _ in range(num_bahias)]  # Empty bays
            for bahia in range(num_bahias):
                num_camiones_bahia = len(camiones_disponibles) // (num_bahias - bahia)
                camiones_asignados = camiones_disponibles[:num_camiones_bahia]
                individuo[bahia] = camiones_asignados
                camiones_disponibles = camiones_disponibles[num_camiones_bahia:]
            poblacion.append(individuo)
        return poblacion

    # Function to calculate the fitness of an individual
    def calcular_fitness(individuo, camiones, productos):
        fitness_total = 0
        tiempo_total = 0
        demanda_satisfecha = [0] * len(productos)
        productos_sobrantes = [0] * len(productos)

        for bahia in individuo:
            tprevio = 0
            for camion_id in bahia:
                camion = camiones[camion_id]
                tiempo_servicio = camion['Pi'] * Tp
                ti = max(camion['Ai'], tprevio)
                fitness_total += camion['prioridad'] * (ti + tiempo_servicio)
                tiempo_total += (tiempo_servicio + ti - camion['Ai'])
                tprevio = ti + tiempo_servicio

                # Update satisfied demand and calculate leftover products
                for i, cantidad_descargada in enumerate(camion['uij']):
                    demanda_satisfecha[i] += cantidad_descargada
                    if demanda_satisfecha[i] > productos[i]['demanda_minima']:
                        productos_sobrantes[i] = demanda_satisfecha[i] - productos[i]['demanda_minima']

        return 1 / (fitness_total + 1), tiempo_total, productos_sobrantes

    # Tournament selection function
    def seleccion(poblacion, fitness):
        seleccionados = []
        for _ in range(tamaño_poblacion):
            torneo = random.sample(list(enumerate(fitness)), 3)
            ganador = max(torneo, key=lambda x: x[1][0])  # Based on fitness
            seleccionados.append(poblacion[ganador[0]])
        return seleccionados

    # Crossover function
    def cruce(padre1, padre2):
        if random.random() < prob_cruce:
            hijo1, hijo2 = [[] for _ in range(num_bahias)], [[] for _ in range(num_bahias)]
            usados_hijo1 = set()
            usados_hijo2 = set()
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
        return padre1, padre2

    # Mutation function
    def mutacion(individuo):
        if random.random() < prob_mutacion:
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

    # Genetic Algorithm
    def algoritmo_genetico(camiones, productos):
        for camion in camiones:
            camion['prioridad'] = calcular_prioridad(camion, productos)

        poblacion = inicializar_poblacion(len(camiones))

        for generacion in range(num_generaciones):
            fitness = [calcular_fitness(individuo, camiones, productos) for individuo in poblacion]
            poblacion_seleccionada = seleccion(poblacion, fitness)

            nueva_poblacion = []
            for i in range(0, tamaño_poblacion, 2):
                padre1, padre2 = poblacion_seleccionada[i], poblacion_seleccionada[i + 1]
                hijo1, hijo2 = cruce(padre1, padre2)
                nueva_poblacion.append(mutacion(hijo1))
                nueva_poblacion.append(mutacion(hijo2))

            poblacion = nueva_poblacion

        fitness = [calcular_fitness(individuo, camiones, productos) for individuo in poblacion]
        mejor_individuo_idx = np.argmax([f[0] for f in fitness])
        mejor_individuo = poblacion[mejor_individuo_idx]
        mejor_tiempo_total = fitness[mejor_individuo_idx][1]
        productos_sobrantes = fitness[mejor_individuo_idx][2]

        return mejor_individuo, mejor_tiempo_total, productos_sobrantes

    # Execute the genetic algorithm
    mejor_solucion, tiempo_total, productos_sobrantes = algoritmo_genetico(camiones, productos)

    # Print the results with truck names
    print("Mejor asignación de camiones a bahías:")
    for bahia_idx, bahia in enumerate(mejor_solucion):
        nombres_camiones = [camiones[camion_id]['nombre'] for camion_id in bahia]
        print(f"Bahía {bahia_idx + 1}: Camiones {', '.join(nombres_camiones)}")

    # Print total time
    print(f"\nTiempo total de descarga: {tiempo_total} minutos")

    # Print remaining products with names
    print("\nProductos sobrantes después de descargar todos los camiones:")
    nombres_productos = productos[0]['nombre']
    for i, sobrante in enumerate(productos_sobrantes):
        print(f"Producto {nombres_productos[i]}: {max(sobrante, 0)} unidades sobrantes")

# Cargar datasets
pending1 = pd.read_csv(f'{output_dir}/pending_camiones_cleanformat.csv')
orders1 = pd.read_csv(f'{output_dir}/orders_cleanformat_resta_existencias.csv')

# Ejecut
algoritmo_genetico(pending1, orders1)
