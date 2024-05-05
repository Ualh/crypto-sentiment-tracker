import requests
import pandas as pd
import datetime
import os

class GlobalEconomyNewsAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://global-economy-news.p.rapidapi.com"
        self.headers = {
            'X-RapidAPI-Key': self.api_key,
            'X-RapidAPI-Host': 'global-economy-news.p.rapidapi.com'
        }

    def fetch_news(self, category, country, query=None, initial=None, final=None):
        """Fetches news articles based on category and country with optional search query and time limits."""
        params = {
            'category': category,
            'country': country,
            'query': query,
            'initial': initial,
            'final': final
        }
        response = requests.get(f"{self.base_url}", headers=self.headers, params=params)
        print(f"Request sent to: {response.url}")  # Debugging
        if response.status_code == 200:
            data = response.json()
            print(f"Received data: {data}")  # Debugging
            return data
        else:
            print(f"Failed to fetch data: {response.status_code}, {response.text}")  # Error details
            return None

    def format_news_data(self, articles):
        """Transforms news data into a pandas DataFrame and standardizes column formats."""
        # Check if 'response' and 'data' keys are present
        if 'response' not in articles or 'data' not in articles['response'] or not articles['response']['data']:
            print("No 'data' key found in articles, or data is empty, received:", articles)
            return pd.DataFrame()  # Return an empty DataFrame if no 'data' key or it's empty

        data = []
        for article in articles['response']['data']:
            try:
                published = pd.to_datetime(article['published']).strftime('%Y-%m-%d %H:%M:%S')
                title = article.get('title', 'No title provided')
                description = article.get('text', 'No description provided')
                data.append({
                    'date': published,
                    'headline': title,
                    'description': description
                })
            except Exception as e:
                print(f"Error processing article {article.get('id', 'Unknown ID')}: {e}, Article Data: {article}")
        
        df = pd.DataFrame(data)
        if df.empty:
            print("DataFrame is created but it's empty. Check the content of 'data'.")
        return df
    
    def save_to_csv(self, df, file_name):
        """Appends news data to an existing CSV file, creating it if it doesn't exist."""
        if df.empty:
            print("No data to save.")
            return
        df.to_csv(file_name, mode='a', header=not os.path.exists(file_name), index=False)
        print(f"Data saved to {file_name}")
