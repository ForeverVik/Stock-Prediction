import streamlit as st
from datetime import date
import yfinance as yf
import numpy as np
np.float_ = np.float64
from prophet.forecaster import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd

#####################################################################
#Website can be ran using the console command 'streamlit run main.py'
#####################################################################

#Loads the data for the chosen stock and caches it so that it doesn't need to be loaded again.
@st.cache_data
def loadData(ticker):
    data = yf.download(ticker, START, TODAY)
    #print(data)
    data.reset_index(inplace=True)
    return data

#Trains the Prophet prediction model based on the chosen stock.
def trainModel(data):
    dfTrain = data[['Date', 'Close']]
    dfTrain = dfTrain.rename(columns={'Date': 'ds', 'Close': 'y'})

    model = Prophet()
    model.fit(dfTrain)
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)

    return model, forecast

#Plots the raw data for the chosen stock.
def plotRawData(ticker):
    data = loadData(ticker)
    
    #Optional dropdown to display tail end of raw data.
    with st.expander(ticker + ' Stock Raw Data', False):
        st.write(data.tail())

    fig = go.Figure()
    #fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock Open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Close'))
    fig.update_traces(showlegend=False)
    fig.layout.update(title_text=ticker+' Stock Graph', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

#Plots the predicted forecast data for the chosen stock.
def plotForecastData(ticker):
    data = loadData(ticker)
    model, forecast = trainModel(data)

    #Optional dropdown to display tail end of raw data.
    with st.expander(ticker + ' Forecast Raw Data', False):
        st.write(forecast.tail())

    fig1 = plot_plotly(model, forecast, ylabel='', xlabel='')
    fig1.update_traces(marker=dict(color='#8dcaff'), mode='lines', selector=dict(type="scatter"))
    fig1.layout.update(title_text=ticker+' Forecast Graph', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig1)

    #Displays forecast components.
    st.subheader("Forecast Components")
    fig2 = model.plot_components(forecast)
    st.write(fig2)

#Main function
if __name__=="__main__":
    #Set how far back to display the stock trends.
    START = '2015-01-01'
    TODAY = date.today()

    st.title('Stock Prediction App')

    #Finds the current stocks in the S&P 500 and lists them as options for prediction modeling.
    stocks = pd.read_html(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    selectedStock = st.selectbox('Select Dataset for Prediction', stocks)
    
    numYears = st.slider('Years of Prediction:', 1, 5)
    period = numYears * 365
    
    plotRawData(selectedStock)
    plotForecastData(selectedStock)