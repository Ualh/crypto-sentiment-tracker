import requests
from datetime import datetime
import pandas as pd
import html
import re
from bs4 import BeautifulSoup
from modules.handlers import DataHandler
import time
import os
from dotenv import load_dotenv

load_dotenv()

class SeekingAlphaNewsAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://seeking-alpha.p.rapidapi.com"
        self.headers = {
            'X-RapidAPI-Key': self.api_key,
            'X-RapidAPI-Host': 'seeking-alpha.p.rapidapi.com'
        }

    def fetch_news(self, category, since=None, until=None, size=20, number=1):
        """
        Fetches news articles from Seeking Alpha based on category and optional time limits.
        
        Parameters:
            category (str): The news category to fetch. Categories include:
                - market-news::all
                - market-news::top-news
                - market-news::on-the-move
                - market-news::market-pulse
                - market-news::notable-calls
                - market-news::buybacks
                - market-news::commodities
                - market-news::crypto
                - market-news::issuance
                - market-news::dividend-stocks
                - market-news::dividend-funds
                - market-news::earnings
                - earnings::earnings-news
                - market-news::global
                - market-news::guidance
                - market-news::ipos
                - market-news::spacs
                - market-news::politics
                - market-news::m-a
                - market-news::us-economy
                - market-news::consumer
                - market-news::energy
                - market-news::financials
                - market-news::healthcare
                - market-news::mlps
                - market-news::reits
                - market-news::technology
            since (int, optional): Unix timestamp for the start of the news period.
            until (int, optional): Unix timestamp for the end of the news period.
            size (int, optional): The number of items per response (max 40).
            number (int, optional): Page index for pagination purposes.

        Returns:
            A JSON response containing the news data if the request is successful; otherwise, None.
        """
        params = {
            'category': category,
            'size': size,
            'number': number
        }
        if since:
            params['since'] = since
        if until:
            params['until'] = until
            
        response = requests.get(f"{self.base_url}/news/v2/list", headers=self.headers, params=params)
        if response.status_code == 200:
            return response.json().get('data', [])
        else:
            print("Failed to fetch data:", response.status_code)
            return []

    def format_news_data(self, news_items):
        """Transforms news data into a pandas DataFrame, parsing dates and cleaning HTML content."""
        data = []
        for item in news_items:
            attributes = item.get('attributes', {})
            title = html.unescape(attributes.get('title', 'No title provided'))

            publish_on = attributes.get('publishOn', '')
            try:
                publish_date = datetime.fromisoformat(publish_on[:-6])  # Remove timezone info for simplicity
            except ValueError:
                publish_date = datetime.now()  # Default to current date/time if parsing fails

            content = html.unescape(attributes.get('content', 'No description provided'))
            content = self.clean_description(content)  # Clean HTML and reduce clutter
            
            data.append({
                'date': publish_date.strftime('%Y-%m-%d %H:%M:%S'),
                'headline': title,
                'description': content
            })

        return pd.DataFrame(data)

    def clean_description(self, text):
        """Cleans HTML tags and filters out invisible content or other unwanted parts."""
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text(separator=" ")  # Use space as separator to avoid words sticking together
        
        # Optional: Remove common unwanted patterns
        text = re.sub(r'Click here to read more', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace

        return text
    
    def fetch_news_by_days(self, days, category):
            dh = DataHandler()
            df_daily_news = pd.DataFrame()
            # days = 2 #for testing
            for i in range(days):
                start = dh.get_date_dt(i + 1)  # Start of day
                end = dh.get_date_dt(i)  # End of day
                initial_unix_s = dh.convert_to_unix_seconds(start)
                final_unix_s = dh.convert_to_unix_seconds(end)

                news_data = self.fetch_news(category=f'market-news::{category}', since=initial_unix_s, until=final_unix_s, size=40)
                time.sleep(1)
                print(news_data)
                if news_data:
                    formatted_data = self.format_news_data(news_data)
                    df_daily_news = pd.concat([df_daily_news, formatted_data], ignore_index=True)
                else:
                    print(f"No news found for the date range starting {dh.get_date_str(i + 1)} to {dh.get_date_str(i)}")

            return df_daily_news
