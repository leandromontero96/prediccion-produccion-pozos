"""
Generador de Datos Sintéticos - Producción de Pozos Petroleros
Simula 100 pozos con 365 días de historia cada uno
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

def generar_datos_pozo(pozo_id, tipo_pozo):
    """Genera datos sintéticos realistas para un pozo petrolero"""

    # Parámetros iniciales según tipo de pozo
    if tipo_pozo == 'Productor Alto':
        prod_inicial = np.random.uniform(800, 1500)  # bbl/día
        presion_inicial = np.random.uniform(3500, 4500)  # psi
        decline_rate = np.random.uniform(0.08, 0.15)  # decline anual
    elif tipo_pozo == 'Productor Medio':
        prod_inicial = np.random.uniform(400, 800)
        presion_inicial = np.random.uniform(2500, 3500)
        decline_rate = np.random.uniform(0.12, 0.20)
    else:  # Productor Bajo
        prod_inicial = np.random.uniform(100, 400)
        presion_inicial = np.random.uniform(1500, 2500)
        decline_rate = np.random.uniform(0.15, 0.25)

    datos = []
    fecha_inicio = datetime(2023, 1, 1)

    for dia in range(365):
        fecha = fecha_inicio + timedelta(days=dia)

        # Decline exponencial + ruido
        decline_factor = np.exp(-decline_rate * (dia / 365))
        produccion = prod_inicial * decline_factor * np.random.uniform(0.85, 1.15)

        # Presión declina con producción
        presion = presion_inicial * decline_factor * np.random.uniform(0.90, 1.05)

        # Water cut aumenta con tiempo
        water_cut = min(95, 10 + (dia / 365) * 40 + np.random.uniform(-5, 5))

        # GOR aumenta al bajar presión
        gor = 500 + (4500 - presion) * 0.3 + np.random.uniform(-50, 50)

        # Temperatura relativamente constante
        temperatura = 180 + np.random.uniform(-10, 10)

        # Choke (válvula) ajustado según producción
        choke = min(64, max(12, 32 + (produccion - 500) / 20))

        datos.append({
            'pozo_id': pozo_id,
            'fecha': fecha,
            'produccion_bbl': max(0, produccion),
            'presion_psi': max(500, presion),
            'water_cut_pct': water_cut,
            'gor_scf_bbl': max(100, gor),
            'temperatura_f': temperatura,
            'choke_64': choke,
            'dias_produccion': dia + 1,
            'tipo_pozo': tipo_pozo
        })

    return datos

# Generar dataset completo
print("Generando datos sinteticos de 100 pozos petroleros...")

todos_datos = []
tipos_pozos = ['Productor Alto'] * 30 + ['Productor Medio'] * 50 + ['Productor Bajo'] * 20

for i, tipo in enumerate(tipos_pozos, 1):
    pozo_id = f"WELL-{i:03d}"
    datos_pozo = generar_datos_pozo(pozo_id, tipo)
    todos_datos.extend(datos_pozo)
    if i % 20 == 0:
        print(f"  - Generados {i} pozos...")

df = pd.DataFrame(todos_datos)

# Crear features de ingeniería
df['mes'] = df['fecha'].dt.month
df['trimestre'] = df['fecha'].dt.quarter
df['produccion_acumulada'] = df.groupby('pozo_id')['produccion_bbl'].cumsum()
df['decline_rate'] = df.groupby('pozo_id')['produccion_bbl'].pct_change() * -1
df['eficiencia'] = df['produccion_bbl'] / (df['presion_psi'] / 1000)

# Guardar
df.to_csv('data/produccion_pozos.csv', index=False)

print(f"\n[OK] Dataset generado exitosamente!")
print(f"   Total registros: {len(df):,}")
print(f"   Total pozos: {df['pozo_id'].nunique()}")
print(f"   Rango fechas: {df['fecha'].min()} a {df['fecha'].max()}")
print(f"\nEstadisticas de Produccion:")
print(df.groupby('tipo_pozo')['produccion_bbl'].agg(['mean', 'min', 'max']).round(2))
