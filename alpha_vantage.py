
#alha_vantage.py
import os
import requests
import json
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

class AlphaVantage:
    def __init__(self, api_key, endpoint):
        self.api_key = api_key
        self.endpoint = endpoint

    def fetch_data(self, url, params):
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.content

        except requests.exceptions.RequestException as e:
            print(f'Error fetching data: {e}')
            return None

    def get_all_alpha_v_tickers(self):
        """
        Fetches a list of all available stock tickers from Alpha Vantage API
        and saves it as a CSV file.

        Returns:
            None
        """
        params = {
            'function': 'LISTING_STATUS',
            'apikey': self.api_key,
        }

        print("Fetching all tickers...")
        response_content = self.fetch_data(self.endpoint, params)
        if response_content:
            csv_filename = f'data/Alpha_Vantage-All_Tickers.csv'
            with open(csv_filename, 'wb') as csv_file:
                csv_file.write(response_content)

            print(f'Data saved to {csv_filename}') # HOW TO SAVE IT TO THE DATA FOLDER ?????!!!!!!
        else:
            print("No data received.")

    def get_ticker_intraday(self, ticker):
        """
        Fetches intraday stock data for a given ticker using the Alpha Vantage API. And saves it into a DataFrame to CSV file.

        Parameters:
            ticker (str): The stock ticker symbol. Shall be in Alpha_Vantage-All_Tickers.csv

        Returns:
            pandas.DataFrame: DataFrame containing intraday stock data.
        """
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': ticker,
            'interval': '1min',
            'outputsize': 'full',
            'apikey': self.api_key,
        }

        response_content = self.fetch_data(self.endpoint, params)
        if response_content:
            try:
                response_json = json.loads(response_content.decode('utf-8'))  # Convert bytes to JSON
                print(response_json)
                data = response_json['Time Series (1min)']

                df = pd.DataFrame(data).T
                df.index = pd.to_datetime(df.index)
                df = df.rename(
                    columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close',
                             '5. volume': 'Volume'})

                # Save DataFrame to CSV file
                csv_filename = f'data/{ticker}_{df.index.min().strftime("%Y-%m-%d-%H_%M")}_to_{df.index.max().strftime("%Y-%m-%d-%H_%M")}.csv'
                df.to_csv(csv_filename)
                print(f'Data saved to {csv_filename}')
                return df

            except (json.JSONDecodeError, KeyError) as e:
                print(f'Error parsing JSON: {e}')
                return None

        else:
            print("No data received.")
            return None


