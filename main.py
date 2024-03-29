# Make sure to activate the virtual environment before running this script
# Activate the environment by running the following command in your terminal:
# source venv\Scripts\activate
from dotenv import load_dotenv
import os
import subprocess
from get_stock_price import StockPriceFetcher
load_dotenv()

#----- Stock Price Fetcher ------
fetcher = StockPriceFetcher('msft')  # Fetch data for Microsoft
fetcher.fetch_price()
#--------------------------------

# Load environment variables from .env file


# Access variables
# TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
# TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET')
# REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
# REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
# CCXT_API_KEY = os.getenv('CCXT_API_KEY')

# # Use the variables, e.g., to configure API clients
# print(TWITTER_API_KEY)