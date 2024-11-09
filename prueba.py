import yfinance as yf 
from datetime import datetime, timedelta 
# Define el símbolo del ticker y el periodo de tiempo 
ticker = '^IBEX' 
start_date = '2012-01-01' 
end_date = datetime.now().strftime('%Y-%m-%d') 
# Obtén los datos de las acciones 
data = yf.download(ticker, start=start_date, end=end_date) 