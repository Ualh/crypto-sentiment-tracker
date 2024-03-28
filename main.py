from dotenv import load_dotenv
import os
import subprocess

# Load environment variables from .env file
load_dotenv()

# Access variables
TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET')
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
CCXT_API_KEY = os.getenv('CCXT_API_KEY')

# Use the variables, e.g., to configure API clients
print(TWITTER_API_KEY)