import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# Define el símbolo del ticker y el periodo de tiempo
ticker = '^IBEX'
start_date = '2012-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')

# Obtén los datos de las acciones
data = yf.download(ticker, start=start_date, end=end_date)

# Guardar en un archivo csv
#data.to_csv('ibex.csv')

# Visualización gráfica
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='IBEX 35', color='b')
plt.title('IBEX 35 - Precio de Cierre')
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre')
plt.legend()
plt.grid(True)
plt.show()

