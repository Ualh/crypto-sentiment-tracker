from dotenv import load_dotenv
import os
import pandas as pd
from flask import Flask, request, render_template, session, redirect, url_for
from flask_caching import Cache
from handlers import DataHandler
from news_api import NewsAPI
from useconomyapi import USEconomyAPI
from seekingalpha import SeekingAlphaNewsAPI
from visualizations import Visualizations
from cryptopanic import CryptoPanicAPI
from cryptonewsapi import CryptoNewsAPI
from coinrankingapi import CryptoDataFetcher
from sentiment import SentimentAnalyzer
import matplotlib
import numpy as np
matplotlib.use('Agg')

hd = DataHandler()
vz = Visualizations()
load_dotenv()

app = Flask(__name__, template_folder='docs', static_folder='static')
# Configure caching, for example, using simple memory cache
app.config["CACHE_TYPE"] = "SimpleCache"
app.config["CACHE_DEFAULT_TIMEOUT"] = 86400  # 24h Cache timeout in seconds
app.config['TESTING'] = True  # True for testing, False for production

app.secret_key = os.urandom(24)  # Or a fixed secret key

cache = Cache(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Store form data to session
        session['days_back'] = request.form['days_back']
        session['category'] = request.form.get('category', 'crypto')
        session['model_type'] = request.form.get('model_type', 'linear')
        return redirect(url_for('dashboard'))
    categories = [
        'defi', 'stablecoin', 'nft', 'dex', 'exchange', 'staking', 'dao', 'meme', 
        'privacy', 'metaverse', 'gaming', 'wrapped', 'layer-1', 'layer-2', 
        'fan-token', 'football-club', 'web3', 'social'
    ]
    return render_template('index.html', categories=categories)


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if request.method == 'POST':
        # Handle form submission for changing lag
        selected_lag_index = int(request.form.get('selectedLag', 0))
        #model_type = request.form.get('model_type', 'linear')
        session['selected_lag_index'] = selected_lag_index
        #session['model_type'] = model_type
        return redirect(url_for('dashboard'))
    else:
        # GET request handling, use the default or session-stored index
        selected_lag_index = session.get('selected_lag_index', 0)
        model_type = session.get('model_type', 'linear')

    days_back = int(session.get('days_back', 1))  # Provide default value if not set
    category = session.get('category', 'crypto')
    
    news_df = fetch_news(days_back)
    price_data, sentiment_data = perform_sentiment_analysis(news_df, days_back, category)
    plot, price_data, sentiment_data = generate_plots(price_data, days_back, sentiment_data)
    combined_data = merge_data(days_back, price_data, sentiment_data)
    
    if model_type == 'linear':
        analysis_results, future_predictions_by_lag = vz.analysis(combined_data, days_back, model_type='linear', for_web=True, predict_days=5)
    else:
        analysis_results, future_predictions_by_lag = vz.analysis(combined_data, days_back, model_type='random_forest', for_web=True, predict_days=5)

    if selected_lag_index >= len(analysis_results):
        selected_lag_index = 0

    future_predictions = future_predictions_by_lag[selected_lag_index] if future_predictions_by_lag else []
    
    # Ensure future_predictions is a list
    if not isinstance(future_predictions, (list, np.ndarray)):
        future_predictions = [future_predictions]

    # Filter out NaN values and calculate the average prediction for each lag
    future_predictions = [pred for pred in future_predictions if not np.isnan(pred)]
    avg_prediction = round(np.mean(future_predictions), 2) if future_predictions else 'N/A'

    # Ensure all entries in future_predictions_by_lag are lists
    all_predictions = []
    for lag_predictions in future_predictions_by_lag:
        if not isinstance(lag_predictions, (list, np.ndarray)):
            lag_predictions = [lag_predictions]
        all_predictions.extend(pred for pred in lag_predictions if not np.isnan(pred))

    # Calculate the overall average prediction across all lags
    overall_avg_prediction = round(np.mean(all_predictions), 2) if all_predictions else 'N/A'

    return render_template('dashboard.html',
                            table=news_df.to_html(classes='display', index=False, table_id='datatablesSimple'),
                            plot=plot,
                            analysis_results=analysis_results,
                            days_back=days_back,
                            category=category,
                            selected_lag_index=selected_lag_index,
                            model_type=model_type,
                            future_predictions=future_predictions,
                            avg_prediction=avg_prediction,
                            overall_avg_prediction=overall_avg_prediction)


#The caching mechanism stores the complete result of the function once it successfully executes
def make_news_cache_key():
    days_back = session.get('days_back', 1)  # Default to 1 if not found
    category = session.get('category', 'crypto')
    return f"fetch_news_{days_back}_{category}"

@cache.cached(timeout=86400, key_prefix=make_news_cache_key)
def fetch_news(days_back):
    print('no cache')
    if app.config['TESTING']:
        # Load mock news data based on 'days_back'
        if days_back == 2:
            return pd.read_csv("data/30d_news.csv")
        elif days_back == 1:
            return pd.read_csv("data/24h_news.csv")
        else:
            raise ValueError("Invalid 'days_back' value. It should be either 1 (daily) or 30 (monthly).")
    else:
    # Determine time frame
        if days_back == 2:
            print('ok, fetching past month news')
            from_date = 30
            # Create an instance of the NewsAPI
            news_api = NewsAPI(api_key=os.getenv('news_api'))
            # Fetch news articles about Crypto from a specific date range
            df = news_api.get_news(topic='crypto', from_date=hd.get_date_str(from_date), to_date=(hd.get_date_str(0)))
            df = hd.standardize_date_format(df, 'date')

        elif days_back == 1:
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

        news_df = pd.concat([df, crypto_news_df_2, crypto_news_df_3])
        hd.process_duplicates(df)
    return news_df



########## Make sentiment Analysis ############
def make_sentiment_analysis_cache_key():
    days_back = session.get('days_back', 1)
    category = session.get('category', 'crypto')
    return f"perform_sentiment_analysis_{days_back}_{category}"

@cache.cached(timeout=86400, key_prefix=make_sentiment_analysis_cache_key)
def perform_sentiment_analysis(news_df, days_back, category):
    print('nothing was cached')
    if app.config['TESTING']:
        if days_back == 2:
            sentiment_data = pd.read_csv("data/30d_news_with_sentiment.csv")
            price_data = pd.read_csv("data/30d_50meme_history.csv")
        elif days_back == 1:
            sentiment_data = pd.read_csv("data/24h_news_with_sentiment.csv")
            price_data = pd.read_csv("data/24h_50meme_history.csv")
        else:
            raise ValueError("Invalid 'days_back' value.")
    else:
        ######## Get Historical Price ########
        api_key = os.getenv('coinranking') 

        if days_back == 2:
            time_period = '30d'
        else:
            time_period = '24h'
        fetcher = CryptoDataFetcher(api_key, default_tags=category, default_limit=2, default_time_period=f"{time_period}")
        price_data = fetcher.fetch_all_history()

        # Initialize SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        sentiment_data = analyzer.add_sentiments_to_df(news_df)
    return price_data, sentiment_data

######### Vizualisations ##########
def generate_plots(price_data, days_back, sentiment_data):
    vz = Visualizations()
    print("days_back:", days_back, "Type of sentiment_data:", type(sentiment_data))
    sentiment_data = vz.average_sentiment_per_time(from_date=days_back, data=sentiment_data)
    price_data = vz.normalize_and_aggregate_prices(price_data)
    plot = vz.plot_normalized_price_and_sentiment(price_data, sentiment_data, for_web=True)
    return plot, price_data, sentiment_data

######### Analysis ##########
def merge_data(days_back, price_data, sentiment_data):
    if days_back == 1:
        price_data['price_change'] = price_data['normalized price'].pct_change().fillna(0) * 100
        price_data['price_change'] = price_data['price_change'].replace([np.inf, -np.inf], np.nan).fillna(0)

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

    elif days_back == 2 or days_back == 30:
        # Get the closing price for each day (last price of the day)
        price_data_daily = price_data['normalized price'].resample('D').last()

        # Calculate the daily price change percentage
        price_data_daily = pd.DataFrame(price_data_daily)  # Ensure it's a DataFrame for the next operations
        price_data_daily['Price Change'] = price_data_daily['normalized price'].pct_change() * 100
        price_data_daily['Price_change'] = price_data['Price_change'].replace([np.inf, -np.inf], np.nan).fillna(0)

        # Shift the price change to align with the day's sentiment to measure its influence on the next day's price change
        price_data_daily['Price Change'] = price_data_daily['Price Change'].shift(-1)

        # Since sentiment is often recorded multiple times a day, we'll average it for daily granularity
        sentiment_data_daily = pd.DataFrame(sentiment_data['average sentiment'].resample('D').mean())

        # Merge the two datasets on the date index
        combined_data = pd.concat([price_data_daily, sentiment_data_daily], axis=1)
        combined_data.columns = ['Normalized Price', 'Next Day Price Change', 'average sentiment']
        combined_data.dropna(inplace=True)  # Drop rows with NaN values that might result from resampling, shifting, or non-overlapping dates
    return combined_data


@app.route('/extra_features')
def extra_features():
    return render_template('extra_features.html')


if __name__ == "__main__":
    app.run(debug=True)
