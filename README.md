# CRYPTO-SENTIMENT-TRACKER

## Description
CRYPTO-SENTIMENT-TRACKER is a Flask web application that integrates various APIs to analyze cryptocurrency news sentiment and its impact on market prices. The app fetches real-time data, performs sentiment analysis, and displays trends through an interactive dashboard.

## Prerequisites
- Python 3.8+
- pip (Python package installer)
- Virtual Environment (recommended)

## Installation

### Clone the Repository
To get started, clone the repository to your local machine:

```
git clone https://github.com/Ualh/crypto-sentiment-tracker.git
```
```
cd yourpathtothegit
```

### Set Up Python Environment

Create and activate a virtual environment to manage dependencies:

```
python -m venv 
```
Activate it (for windows venv/bin/activate for mac) 
```
venv\Scripts\activate
```

Install the required packages:

```
pip install -r requirements.txt
```

### Configure Environment Variables
Set up the necessary API keys by creating a `.env` file in the root directory with the following content:

```
news_api=<your_newsapi_key>
seeking_alpha=<your_seekingalphaapi_key>
coinranking=<your_coinrankingapi_key>
useconomyapi=<your_useconomyapi_key>
cryptonewsapi=<your_cryptonewsapi_key>
cryptopanic=<your_cryptopanic_key>
```

Replace each placeholder with your actual API keys. These are crucial for the application's data fetching functionalities.

## Running the Application
To launch the application, run the following command:

```
flask run
```

This will start the server on http://127.0.0.1:5000/. Navigate to this URL in a web browser to access the application interface.

## Features
- **Home Page:** Users can select cryptocurrency categories and analysis models. Submit the form to generate insights.
- **Dashboard:** Interactive charts and sentiment analysis results are displayed based on user-selected parameters.
- **Extra Features:** Access to additional analytical tools and models.

## Testing
Be sure to change False to True in `app.py`

```
app.config['TESTING'] = False  # True for testing, False for production
```
Be sure to implement the data you want in there :

```
@cache.cached(timeout=86400, key_prefix=make_news_cache_key)
def fetch_news(days_back):
    print('no cache')
    if app.config['TESTING']:
        # Load mock news data based on 'days_back'
        if days_back == 2:
            return pd.read_csv("data/2024-04-05_30d_news.csv")
        elif days_back == 1:
            return pd.read_csv("data/2024-04-05_24h_news.csv")
        else:
            raise ValueError("Invalid 'days_back' value. It should be either 1 (daily) or 30 (monthly).")
   
```

Go into the visualisation class, `average_sentiment_per_time` function and change the days so that it correspond to the days back from today, and the date the data was fetched (in the csv)

```
        elif from_date == 1:
            # Get data from the last 2 days
            end_date = pd.Timestamp.today()
            start_date = end_date - pd.Timedelta(days= 1.5)
        elif from_date in [30, 2]:
            data['date'] = data['date'].dt.round('h')
            # Get data from the last 30 days
            end_date = pd.Timestamp.today()
            start_date = end_date - pd.Timedelta(days=31)
```

## Adjustements

- Selecting number of coins to aggergate based ont the category : 
        Go to `modules`> `coinrankingapi.py` > line 223
        change `default_limit=50` to 2 or any number of coin you want to aggregate.

## Troubleshooting
If you encounter any issues:
- Check that all prerequisites are properly installed.
- Verify that the `.env` file contains correct API keys.
- Make sure that all dependencies are installed via `requirements.txt`.


## License
This project is licensed under the MIT License. See the LICENSE file for more details.
