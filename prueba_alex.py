import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Configuración inicial
ticker = '^IBEX'
start_date = '2012-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')

# Descargar los datos del índice IBEX
data = yf.download(ticker, start=start_date, end=end_date)
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Llenar valores faltantes y eliminar outliers
data = data.interpolate(method='linear')

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]

for col in data.columns:
    data = remove_outliers(data, col)

# Normalizar los datos
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Convertir a DataFrame escalado para facilitar el trabajo con las columnas
scaled_data = pd.DataFrame(scaled_data, columns=['Open', 'High', 'Low', 'Close', 'Volume'])

# Crear secuencias para el modelo LSTM
sequence_length = 60  # Número de días en cada secuencia
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length].values)
        targets.append(data['Close'].iloc[i+seq_length])
    return np.array(sequences), np.array(targets)

X, y = create_sequences(scaled_data, sequence_length)

# Dividir los datos en entrenamiento y prueba
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Construcción del modelo LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.1, verbose=1)

# Generar predicciones
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], 4))), axis=1))[:, 0]

# Calcular métricas
y_test_true = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 4))), axis=1))[:, 0]
mse = np.mean((predictions - y_test_true) ** 2)
mae = np.mean(np.abs(predictions - y_test_true))
rmse = np.sqrt(mse)

print(f"MSE: {mse}, MAE: {mae}, RMSE: {rmse}")

# Visualización de predicciones
plt.figure(figsize=(14, 5))
plt.plot(y_test_true, label="Valor Real", color='blue')
plt.plot(predictions, label="Predicción", color='red')
plt.xlabel("Días")
plt.ylabel("Precio")
plt.title("Predicciones de Precios del IBEX 35")
plt.legend()
plt.show()

# Visualización de la pérdida del modelo
plt.figure(figsize=(14, 5))
plt.plot(history.history['loss'], label='Training Loss', color='green')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.title("Pérdida del Modelo durante el Entrenamiento")
plt.legend()
plt.show()
