import requests
import pandas as pd
import os

class USEconomyAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://us-economy-news.p.rapidapi.com"
        self.headers = {
            'X-RapidAPI-Key': self.api_key,
            'X-RapidAPI-Host': 'us-economy-news.p.rapidapi.com'
        }

    def fetch_news(self, category, query=None, initial=None, final=None):
        """Fetches news articles based on category with optional search query and time limits."""
        params = {
            'category': category,
            'query': query,
            'initial': initial,
            'final': final
        }
        response = requests.get(self.base_url, headers=self.headers, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to fetch data: {response.status_code}, {response.text}")

    def format_news_data(self, articles):
        """Transforms news data into a pandas DataFrame and standardizes column formats."""
        if 'data' not in articles['response']:
            return pd.DataFrame()  # Return an empty DataFrame if no 'data' key

        data = []
        for article in articles['response']['data']:
            data.append({
                'date': pd.to_datetime(article['published']).strftime('%Y-%m-%d %H:%M:%S'),
                'headline': article.get('title', 'No title provided'),
                'description': article.get('text', 'No description provided')
            })
        return pd.DataFrame(data)

    def save_to_csv(self, df, file_name):
        """Appends news data to an existing CSV file, creating it if it doesn't exist."""
        if not os.path.exists(file_name):
            df.to_csv(file_name, index=False)
        else:
            df.to_csv(file_name, mode='a', header=False, index=False)
