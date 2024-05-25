import requests
import pandas as pd
import os

class CryptoNewsAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://cryptocurrency-news2.p.rapidapi.com/v1/"
        self.headers = {
            'X-RapidAPI-Key': self.api_key,
            'X-RapidAPI-Host': 'cryptocurrency-news2.p.rapidapi.com'
        }

    def fetch_news(self, source):
        """Fetches cryptocurrency news articles from specified source."""
        url = f"{self.base_url}{source}"
        response = requests.get(url, headers=self.headers)
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
        if 'data' not in articles:
            print("No 'data' key found in articles, or data is empty, received:", articles)
            return pd.DataFrame()  # Return an empty DataFrame if no 'data' key or it's empty

        data = []
        for article in articles['data']:
            try:
                published = pd.to_datetime(article['createdAt']).strftime('%Y-%m-%d %H:%M:%S')
                title = article.get('title', 'No title provided').replace(',', '|')
                description = article.get('description', 'No description provided').replace(',', '|')
                data.append({
                    'date': published,
                    'headline': title,
                    'description': description,
                })
            except Exception as e:
                print(f"Error processing article: {e}, Article Data: {article}")

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

