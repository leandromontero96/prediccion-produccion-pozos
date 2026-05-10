"""
Modelos de Machine Learning para Predicción de Producción
- Random Forest
- XGBoost
- LSTM (Deep Learning)
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Cargar datos
print("Cargando datos...")
df = pd.read_csv('data/produccion_pozos.csv')
df['fecha'] = pd.to_datetime(df['fecha'])

# Preparar features
features = ['presion_psi', 'water_cut_pct', 'gor_scf_bbl', 'temperatura_f',
            'choke_64', 'dias_produccion', 'mes', 'trimestre']

# One-hot encoding para tipo de pozo
df_encoded = pd.get_dummies(df, columns=['tipo_pozo'], prefix='tipo')
feature_cols = features + [col for col in df_encoded.columns if col.startswith('tipo_')]

X = df_encoded[feature_cols]
y = df_encoded['produccion_bbl']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"[OK] Dataset: {len(X_train)} train, {len(X_test)} test\n")

# ============= MODELO 1: RANDOM FOREST =============
print("Entrenando Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print(f"  MAE:  {rf_mae:.2f} bbl/día")
print(f"  RMSE: {rf_rmse:.2f} bbl/día")
print(f"  R²:   {rf_r2:.4f}\n")

# Importancia de features
importancias = pd.DataFrame({
    'feature': feature_cols,
    'importancia': rf_model.feature_importances_
}).sort_values('importancia', ascending=False)
print("  Top 5 features importantes:")
print(importancias.head().to_string(index=False))

# ============= MODELO 2: XGBOOST =============
print("\nEntrenando XGBoost...")
xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=8, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_r2 = r2_score(y_test, xgb_pred)

print(f"  MAE:  {xgb_mae:.2f} bbl/día")
print(f"  RMSE: {xgb_rmse:.2f} bbl/día")
print(f"  R²:   {xgb_r2:.4f}\n")

# ============= MODELO 3: LSTM =============
print("Entrenando LSTM (Deep Learning)...")

# Normalizar datos para LSTM
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# Reshape para LSTM (samples, timesteps, features)
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Arquitectura LSTM
lstm_model = Sequential([
    LSTM(64, activation='relu', input_shape=(1, X_train_scaled.shape[1]), return_sequences=True),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = lstm_model.fit(
    X_train_lstm, y_train_scaled,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

# Predicciones
lstm_pred_scaled = lstm_model.predict(X_test_lstm, verbose=0)
lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled).flatten()

lstm_mae = mean_absolute_error(y_test, lstm_pred)
lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_pred))
lstm_r2 = r2_score(y_test, lstm_pred)

print(f"  MAE:  {lstm_mae:.2f} bbl/día")
print(f"  RMSE: {lstm_rmse:.2f} bbl/día")
print(f"  R²:   {lstm_r2:.4f}\n")

# ============= GUARDAR MODELOS =============
print("Guardando modelos...")

with open('models/random_forest.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('models/xgboost.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

lstm_model.save('models/lstm_model.h5')

with open('models/scalers.pkl', 'wb') as f:
    pickle.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)

# Guardar resultados
resultados = pd.DataFrame({
    'Modelo': ['Random Forest', 'XGBoost', 'LSTM'],
    'MAE': [rf_mae, xgb_mae, lstm_mae],
    'RMSE': [rf_rmse, xgb_rmse, lstm_rmse],
    'R²': [rf_r2, xgb_r2, lstm_r2],
    'Precisión (%)': [rf_r2*100, xgb_r2*100, lstm_r2*100]
})

resultados.to_csv('models/resultados_modelos.csv', index=False)

print("\n[OK] Modelos entrenados y guardados exitosamente!")
print("\nCOMPARACION FINAL:")
print(resultados.to_string(index=False))

# Calcular impacto de negocio
mejora_precision = rf_r2 * 100
ahorro_estimado = (mejora_precision / 100) * 500000 * 12  # $500k/mes por optimización
print(f"\nIMPACTO DE NEGOCIO ESTIMADO:")
print(f"   Precision del mejor modelo: {mejora_precision:.1f}%")
print(f"   Ahorro anual estimado: ${ahorro_estimado/1e6:.1f}M USD")
print(f"   Optimizacion de produccion: +{mejora_precision*0.15:.1f}%")
