import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests
load_dotenv()
apikey = os.getenv('apikey_marketstack')
apiendpoint = 'http://api.marketstack.com/v1/tickers/aapl/eod'

params = {
    'access_key': apikey,
}

response = requests.get(apiendpoint, params=params)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    # Process the data as needed
    print(data)
else:
    print("Request failed with status code:", response.status_code)
    print("Response content:", response.text)