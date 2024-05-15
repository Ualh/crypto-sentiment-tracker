from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import pandas as pd

from handlers import DataHandler
from news_api import NewsAPI
from useconomyapi import USEconomyAPI
from seekingalpha import SeekingAlphaNewsAPI
from visualizations import Visualizations
from cryptopanic import CryptoPanicAPI
from cryptonewsapi import CryptoNewsAPI
from coinrankingapi import CryptoDataFetcher
from sentiment import SentimentAnalyzer

hd = DataHandler()
vz = Visualizations()


# Load environment variables
load_dotenv()

######## Get news article ########
hd = DataHandler()

input = int(input("how many days back do you want ? Either 2 for monthly or 1 for daily: "))
if input == 2:
    print('ok, fetching past month news')
    from_date = 30
    # Create an instance of the NewsAPI
    news_api = NewsAPI(api_key=os.getenv('news_api'))
    # Fetch news articles about Crypto from a specific date range
    df = news_api.get_news(topic='crypto', from_date=hd.get_date_str(from_date), to_date=(hd.get_date_str(0)))
    df = hd.standardize_date_format(df, 'date')

elif input == 1:
    print('ok, fetching latest news')
    from_date = 1
    auth_token = os.getenv('cryptopanic')  # Example token
    crypto_panic_api = CryptoPanicAPI(auth_token)
    filters = {
        'public': 'true',
        'filter': 'hot',
        'currencies': None
    }
    df = crypto_panic_api.format_news_data(crypto_panic_api.fetch_news(filters))
    
    api_key = os.getenv('cryptonewsapi')
    crypto_news_api = CryptoNewsAPI(api_key)
    sources = ['coindesk', 'cointelegraph', 'bitcoinist', 'decrypt', 'bsc', 'theguardian']
    all_news = []  # List to store data from each source
    for source in sources:
        news_data = crypto_news_api.fetch_news(source) 
        if news_data:  # Ensure there is data before formatting
            formatted_data = crypto_news_api.format_news_data(news_data)
            if not formatted_data.empty:
                all_news.append(formatted_data)  # Append the formatted DataFrame to the list

    # Concatenate all dataframes if all_news is not empty
    if all_news:
        df_daily_news_2 = pd.concat(all_news)
        df = pd.concat([df,df_daily_news_2])

else:
    raise ValueError("Invalid input for days back. Choose 1 for daily or 2 for monthly.")

api = USEconomyAPI(os.getenv('useconomyapi'))
starting_date = hd.convert_to_unix_ms(hd.get_date_dt(from_date))
ending_date = hd.convert_to_unix_ms(hd.get_date_dt(0))
crypto_news_df_2 = api.fetch_news(category='economy', initial=starting_date, final=ending_date )
crypto_news_df_2 = api.format_news_data(crypto_news_df_2)

api_key = os.getenv('seeking_alpha')
seeking_alpha_api = SeekingAlphaNewsAPI(api_key)
crypto_news_df_3 = seeking_alpha_api.fetch_news_by_days(from_date, 'crypto')


######## Get Historical Price ########
category = input("For which category do you want to analyse the sentiment ? ")
from coinrankingapi import CryptoDataFetcher
api_key = os.getenv('coinranking') 

if from_date == 30:
    time_period = '30d'
else:
    time_period = '24h'

fetcher = CryptoDataFetcher(api_key, default_tags="meme", default_limit=2, default_time_period=f"{time_period}")
price_data = fetcher.fetch_all_history()

########## Make sentiment Analysis ############
df = pd.concat([df, crypto_news_df_2, crypto_news_df_3])
hd.process_duplicates(df)
# Initialize SentimentAnalyzer
analyzer = SentimentAnalyzer()
sentiment_data = analyzer.add_sentiments_to_df(df)

######### Vizualisations ##########

vz = Visualizations()
sentiment_data = vz.average_sentiment_per_time(sentiment_data)
price_data = vz.normalize_and_aggregate_prices(price_data)
plot = vz.plot_normalized_price_and_sentiment(price_data, sentiment_data)

######## Analysis ########
if from_date == 1 : 
    # Calculate percentage price change directly from normalized prices
    price_data['price_change'] = price_data['normalized price'].pct_change() * 100
    price_data['price_change'] = price_data['price_change'].fillna(0)  # Replace NaN values with 0

    # Resample price data to 5-minute intervals, forward filling the last known prices and changes
    price_data_resampled = price_data.resample('5min').last().ffill()

    # Round sentiment data timestamps to the nearest 5 minutes
    sentiment_data.index = sentiment_data.index.round('5min')

    # Merge using merge_asof to align sentiment data with the nearest price data
    combined_data = pd.merge_asof(sentiment_data.sort_index(), price_data_resampled.reset_index(), 
                                left_index=True, right_on='timestamp', direction='forward')

    # Since we need the next period's price change, shift the 'price_change' column by -1
    combined_data['Next 5min Price Change'] = combined_data['price_change'].shift(-1)

    # Rename columns to match function expectations
    combined_data.rename(columns={'Average Sentiment': 'Average Sentiment'}, inplace=True)

    # Drop the 'timestamp' and original 'price_change' columns if not needed
    combined_data.drop(columns=['timestamp', 'price_change'], inplace=True)

    combined_data
elif from_date == 30:
    # Get the closing price for each day (last price of the day)
    price_data_daily = price_data['normalized price'].resample('D').last()

    # Calculate the daily price change percentage
    price_data_daily = pd.DataFrame(price_data_daily)  # Ensure it's a DataFrame for the next operations
    price_data_daily['Price Change'] = price_data_daily['normalized price'].pct_change() * 100

    # Shift the price change to align with the day's sentiment to measure its influence on the next day's price change
    price_data_daily['Price Change'] = price_data_daily['Price Change'].shift(-1)

    # Since sentiment is often recorded multiple times a day, we'll average it for daily granularity
    sentiment_data_daily = pd.DataFrame(sentiment_data['Average Sentiment'].resample('D').mean())

    # Merge the two datasets on the date index
    combined_data = pd.concat([price_data_daily, sentiment_data_daily], axis=1)
    combined_data.columns = ['Normalized Price', 'Next Day Price Change', 'Average Sentiment']
    combined_data.dropna(inplace=True)  # Drop rows with NaN values that might result from resampling, shifting, or non-overlapping dates

vz.analysis(combined_data, from_date)