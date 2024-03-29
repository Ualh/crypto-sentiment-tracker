# Make sure to activate the virtual environment before running this script
# Activate the environment by running the following command in your terminal:
# source venv\Scripts\activate
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests

class StockPriceFetcher:
    def __init__(self, ticker):
        load_dotenv()
        self.apikey = os.getenv('apikey_marketstack')
        self.apiendpoint = f'http://api.marketstack.com/v1/tickers/{ticker}/eod'

    def fetch_price(self):
        params = {
            'access_key': self.apikey,
        }

        response = requests.get(self.apiendpoint, params=params)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            # Process the data as needed
            print(data)
        else:
            print("Request failed with status code:", response.status_code)
            print("Response content:", response.text)