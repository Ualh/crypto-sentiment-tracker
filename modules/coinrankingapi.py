import requests
import pandas as pd
import matplotlib.pyplot as plt
import csv
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates
import numpy as np

class CryptoDataFetcher:
    def __init__(self, api_key, default_tags="meme", default_limit=50, default_time_period="30d"):
        """
        Initializes the CryptoDataFetcher with default settings for fetching coin data.
        
        Args:
        - api_key (str): Your API key for the Coinranking API.
        - default_tags (str): Default tags to filter coins. Common tags include 'meme', 'defi', 'nft', etc.
        - default_limit (int): Default number of coin entries to fetch. Maximum often depends on your API plan.
        - default_time_period (str): Default time period for fetching historical data. Options include '24h', '7d', '30d', '3m', '1y', etc.
        """
        self.api_key = api_key
        self.base_url = "https://coinranking1.p.rapidapi.com"
        self.headers = {
            'X-RapidAPI-Key': self.api_key,
            'X-RapidAPI-Host': 'coinranking1.p.rapidapi.com'
        }
        self.default_tags = default_tags
        self.default_limit = default_limit
        self.default_time_period = default_time_period

    def send_request(self, url, params):
        """
        Sends a GET request to the specified URL with the given parameters.

        Args:
            url (str): The URL to send the request to.
            params (dict): Dictionary containing query parameters.

        Returns:
            Response: The response object.
        """
        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code != 200:
            print(f"Response: {response.text}")
        return response
    
    def get_coin_uuids(self, tags=None, limit=None):
        """
        Fetches UUIDs for coins based on specified tags and limit.
        
        Args:
        - 
        - tags (str): Tags to filter coins, such as defi, stablecoin, nft, dex, exchange, staking, dao, meme, privacy...
        - limit (int): The number of coin entries to fetch. Maximum often depends on your API plan.

        Returns:
        - dict: A dictionary mapping coin names to their UUIDs.
        """
        if tags is None:
            tags = self.default_tags
        if limit is None:
            limit = self.default_limit
        url = f"{self.base_url}/coins"
        querystring = {"tags": tags, "limit": str(limit)}
        response = requests.get(url, headers=self.headers, params=querystring)
        data = response.json()
        if data['status'] == 'success':
            return {coin['name']: coin['uuid'] for coin in data['data']['coins']}
        else:
            print(f"Failed to fetch coin UUIDs with tags {tags}")
            return {}

    def get_coin_history(self, uuid, time_period=None):
        """
        Fetches historical price data for a given coin UUID over a specified time period.
        
        Args:
        - uuid (str): The UUID of the coin.
        - time_period (str): Timeperiod where the change and history are based on Default value: 24h Allowed values: 1h 3h 12h 24h 7d 30d 3m 1y 3y 5y

        Returns:
        - list: A list of historical data entries.
        """
        if time_period is None:
            time_period = self.default_time_period
        url = f"{self.base_url}/coin/{uuid}/history"
        querystring = {"timePeriod": time_period}
        try:
            response = self.send_request(url, querystring)
            data = response.json()
            if data['status'] == 'success':
                return data['data']['history']
            else:
                print(f"Failed to fetch history for UUID: {uuid}")
                return []
        except Exception as e:
            print(f"Error fetching history data: {e}")
            return []

    def fetch_all_history(self, tags=None, limit=None, time_period=None):
        """
        Fetches all historical data for coins based on specified tags, limit, and time period.
        
        Args:
        - tags (str): Tags to filter coins.
        - limit (int): The number of coin entries to fetch.
        - time_period (str): Time period for the history data.

        Returns:
        - DataFrame: A pandas DataFrame containing the historical data for the filtered coins.
        """
        coins = self.get_coin_uuids(tags, limit)
        all_data = []
        for name, uuid in coins.items():
            print(f"Fetching historical data for {name} with UUID {uuid}...")
            history = self.get_coin_history(uuid, time_period)
            if history:
                for entry in history:
                    try:
                        if entry['price'] is not None:
                            price = float(entry['price'])
                            all_data.append({
                                'coin name': name,
                                'timestamp': entry['timestamp'],
                                'price': price
                            })
                        else:
                            print(f"Skipping entry with None price: {entry}")
                    except Exception as e:
                        print(f"Error processing entry {entry}: {e}")
        return pd.DataFrame(all_data)


    def write_to_csv(self, data, filename=f'data/history.csv'):
        """
        Writes historical data to a CSV file.
        
        Args:
        - data (DataFrame): The historical data to write.
        - filename (str): The filename for the CSV file.

        Prints the location of the CSV file.
        """
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['coin name', 'timestamp', 'price'])
            for _, row in data.iterrows():
                writer.writerow([row['coin name'], row['timestamp'], row['price']])
        print(f"Data has been written to {filename}")
