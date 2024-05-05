import requests
import pandas as pd
from datetime import datetime

class CryptoPanicAPI:
    def __init__(self, auth_token):
        """
        Initializes the CryptoPanicAPI with the necessary authentication token.

        Args:
            auth_token (str): Your API authentication token provided by CryptoPanic.
        """
        self.auth_token = auth_token
        self.base_url = "https://cryptopanic.com/api/v1/posts/"
        self.headers = {
            'Content-Type': 'application/json'
        }

    def fetch_news(self, filters=None):
        """
        Fetches news posts from CryptoPanic based on specified filters.

        Args:
            filters (dict, optional): A dictionary containing API parameters as key-value pairs. Examples of keys can be:
                - 'filter': Filters the posts by categories like filter=(rising|hot|bullish|bearish|important|saved|lol)
                - 'public': Set to 'true' to use the public API.
                - 'currencies': Filter news by specific currencies like 'BTC,ETH', currencies=CURRENCY_CODE1,CURRENCY_CODE2 (max 50).
                - 'regions': Filter news by specific regions like 'en,de', regions=REGION_CODE1,REGION_CODE2. Default: en. Available regions: en (English), de (Deutsch), nl (Dutch), es (Español), fr (Français), it (Italiano), pt (Português), ru (Русский), tr (Türkçe), ar (عربي), cn (中國人), jp (日本), ko (한국인)
                - 'kind': Specify the kind of posts to retrieve, e.g., 'news'. Default: all. Available values: news or media

        Returns:
            list: A list of news posts if the request is successful; otherwise, an empty list.
        """
        params = {
            'auth_token': self.auth_token
        }
        if filters:
            params.update(filters)

        response = requests.get(self.base_url, headers=self.headers, params=params)
        if response.status_code == 200:
            data = response.json()
            return data['results'] if 'results' in data else []
        else:
            print(f"Failed to fetch data: {response.status_code}, {response.text}")
            return []

    def format_news_data(self, news_items):
        """
        Transforms news data into a pandas DataFrame and standardizes column formats.

        Args:
            news_items (list): A list of dictionaries containing news data.

        Returns:
            DataFrame: A pandas DataFrame containing formatted news data.
        """
        data = []
        for item in news_items:
            published = datetime.fromisoformat(item['published_at']).strftime('%Y-%m-%d %H:%M:%S')
            title = item.get('title', 'No title provided')
            description = item.get('description', 'No description provided')
            if description == 'No description provided':
                description = title  # Copy title to description if the default text is present
            data.append({
                'date': published,
                'headline': title,
                'description': description
            })

        return pd.DataFrame(data)
