#news_api.py
import os
import requests
import json
from dotenv import load_dotenv
load_dotenv()

class NewsAPI:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('news_api')
        self.base_url = "https://newsapi.org/v2"

    def make_request(self, endpoint, **params):
        """
        Constructs and sends a request to the specified endpoint.

        Endpoints:
        News API has 2 main endpoints:

        1. Everything /v2/everything – search every article published by over 80,000 different sources large and small in the last 5 years. This endpoint is ideal for news analysis and article discovery.

        2. Top headlines /v2/top-headlines – returns breaking news headlines for countries, categories, and singular publishers. This is perfect for use with news tickers or anywhere you want to use live up-to-date news headlines.

        There is also a minor endpoint that can be used to retrieve a small subset of the publishers we can scan:

        3. Sources /v2/top-headlines/sources – returns information (including name, description, and category) about the most notable sources available for obtaining top headlines from. This list could be piped directly through to your users when showing them some of the options available.
        """
        url = f"{self.base_url}/{endpoint}"
        params['apiKey'] = self.api_key
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raises a HTTPError if the response code was not 200
            return response.json()  # Directly return the JSON response
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None
        
    def get_sources(self, category=None, language=None, country=None):
        """Retrieve news sources, optionally filtered by category or country."""
        params = {}
        if category:
            params['category'] = category
        if language:
            params['language'] = language
        if country:
            params['country'] = country
        
        return self.make_request("top-headlines/sources", **params)
    
    def get_news(self, topic, from_date, to_date):
        """Fetches news headlines related to a given topic."""
        params = {
            'q': topic,
            'source': 'bloomberg, fortune, the-wall-street-journal',
            'domains': 'bloomberg.com, fortune.com, wsj.com',
            'from': from_date,
            'to': to_date,
            'language': 'en',
            'sortBy': 'relevancy',
        }
        response = self.make_request("everything", **params)
        if response and response.get('status') == 'ok':
            articles = response.get('articles', [])
            headlines = [f'Headline from: {article["source"]["name"]}\n{article["title"]}\n{article["description"]}\n{article["url"]}\n' for article in articles[:3]]
            print(f'Found {response['totalResults']} articles')
            return headlines
        else:
            print(f"No data or error occurred for topic: {topic}")
            return []
