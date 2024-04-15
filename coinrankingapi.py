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

    def get_coin_uuids(self, tags=None, limit=None):
        """
        Fetches UUIDs for coins based on specified tags and limit.
        
        Args:
        - tags (str): Tags to filter coins, such as 'meme', 'defi', 'nft', 'stablecoin', 'privacy', etc.
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
        - time_period (str): Time period for the history data. Options include '24h', '7d', '30d', '3m', '1y', etc.

        Returns:
        - list: A list of historical data entries.
        """
        if time_period is None:
            time_period = self.default_time_period
        url = f"{self.base_url}/coin/{uuid}/history"
        querystring = {"timePeriod": time_period}
        response = requests.get(url, headers=self.headers, params=querystring)
        data = response.json()
        if data['status'] == 'success':
            return data['data']['history']
        else:
            print(f"Failed to fetch history for UUID: {uuid}")
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
            print(f"Fetching historical data for {name}...")
            history = self.get_coin_history(uuid, time_period)
            for entry in history:
                all_data.append({
                    'coin': name,
                    'timestamp': entry['timestamp'],
                    'price': float(entry['price'])
                })
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
            writer.writerow(['Coin Name', 'Timestamp', 'Price'])
            for _, row in data.iterrows():
                writer.writerow([row['coin'], row['timestamp'], row['price']])
        print(f"Data has been written to {filename}")

    def plot_aggregated_prices(self, data, sentiment_data=None, window_size=7):
        """
        Aggregates, smooths, and plots normalized historical price data, along with sentiment data if provided.

        Args:
        - data (DataFrame): DataFrame containing at least 'timestamp' and 'price' columns.
        - sentiment_data (DataFrame): Optional DataFrame containing 'date' and 'sentiment' columns.
        - window_size (int): Window size for smoothing the sentiment data.
        """
        data.columns = data.columns.str.lower()
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
        data.set_index('timestamp', inplace=True)

        # Normalize the price data
        scaler = MinMaxScaler()
        data['normalized_price'] = scaler.fit_transform(data[['price']])
        aggregated_data = data.groupby(data.index)['normalized_price'].mean().reset_index()

        fig, ax1 = plt.subplots(figsize=(12, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, 10))

        # Plot price data
        ax1.plot(aggregated_data['timestamp'], aggregated_data['normalized_price'], label='Normalized Average Price', color=colors[0])
        ax1.set_title('Normalized Price and Sentiment Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Normalized Price (0 to 1 Scale)', color=colors[0])
        ax1.tick_params(axis='y', labelcolor=colors[0])

        if sentiment_data is not None:
            sentiment_data.columns = sentiment_data.columns.str.lower()
            sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
            sentiment_data.set_index('date', inplace=True)
            sentiment_data.sort_index(inplace=True)

            # Apply smoothing
            sentiment_data['smoothed_sentiment'] = sentiment_data['sentiment'].rolling(window=window_size, min_periods=1).mean()

            # Plot smoothed sentiment data
            ax2 = ax1.twinx()
            ax2.plot(sentiment_data.index, sentiment_data['smoothed_sentiment'], label='Smoothed Sentiment', color=colors[5], linestyle='--')
            ax2.set_ylabel('Sentiment (0 to 1 Scale)', color=colors[5])
            ax2.tick_params(axis='y', labelcolor=colors[5])

        fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        plt.grid(True)
        plt.show()

        return fig

