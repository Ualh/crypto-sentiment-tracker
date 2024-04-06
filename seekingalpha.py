from html import unescape
import re
import pandas as pd
import requests

class CryptoNewsSentiment:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://seeking-alpha.p.rapidapi.com/news/v2/list"
        self.headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }
        
    def clean_html(self, raw_html):
        """
        Utility function to clean HTML tags from a string.
        """
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', raw_html)
        return unescape(cleantext)
        
    def get_crypto_news(self, since=None, until=None, size=40, number=1):
        """
        Fetches crypto market news.

        Parameters:
        - since: Unix timestamp for the start of the date range.
        - until: Unix timestamp for the end of the date range.
        - size: Number of items per response.
        - number: Page index for paging.

        Returns: A DataFrame containing cleaned crypto market news.
        """
        querystring = {
            "category": "market-news::crypto",
            "size": str(size),
            "number": str(number)
        }
        
        if since is not None:
            querystring["since"] = since
        if until is not None:
            querystring["until"] = until
            
        response = requests.get(self.base_url, headers=self.headers, params=querystring)
        
        if response.status_code == 200:
            try:
                news_data = response.json().get('data', [])
                
                news_list = []
                for item in news_data:
                    publish_date = pd.to_datetime(item['attributes']['publishOn']).strftime('%Y-%m-%d %H:%M:%S')
                    title = item['attributes']['title']
                    content = self.clean_html(item['attributes']['content'])
                    news_list.append({'date': publish_date, 'headline': title, 'description': content})
                
                news_df = pd.DataFrame(news_list, columns=['date', 'headline', 'description'])
                return news_df
            except KeyError as e:
                print(f"Key error: {e}")
        else:
            # Check if the status code indicates a rate limit has been exceeded
            if response.status_code == 429:
                print(f"Error: Rate limit exceeded for Seek Alpha API :(. \n{response.status_code}: {response.reason} ")
            # You can add more elif statements here for other specific status codes if needed
            else:
                # Generic error message for other types of errors
                print(f"Failed to fetch news. Status code: {response.status_code}, Reason: {response.reason}")

        
        # Return an empty DataFrame if the request failed or parsing was unsuccessful
        return pd.DataFrame(columns=['date', 'headline', 'description'])
