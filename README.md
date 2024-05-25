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

git clone https://github.com/yourusername/CRYPTO-SENTIMENT-TRACKER.git
cd CRYPTO-SENTIMENT-TRACKER


### Set Up Python Environment

Create and activate a virtual environment to manage dependencies:

python -m venv venv
source venv/bin/activate # On Windows use venv\Scripts\activate

```
Install the required packages:
pip install -r requirements.txt
```

### Configure Environment Variables
Set up the necessary API keys by creating a `.env` file in the root directory with the following content:

news_api=<your_newsapi_key>
seeking_alpha=<your_seekingalphaapi_key>
coinranking=<your_coinrankingapi_key>
useconomyapi=<your_useconomyapi_key>
cryptonewsapi=<your_cryptonewsapi_key>
cryptopanic=<your_cryptopanic_key>

Replace each placeholder with your actual API keys. These are crucial for the application's data fetching functionalities.

## Running the Application
To launch the application, run the following command:

flask run

This will start the server on http://127.0.0.1:5000/. Navigate to this URL in a web browser to access the application interface.

## Features
- **Home Page:** Users can select cryptocurrency categories and analysis models. Submit the form to generate insights.
- **Dashboard:** Interactive charts and sentiment analysis results are displayed based on user-selected parameters.
- **Extra Features:** Access to additional analytical tools and models.

## Testing
Execute the following command to run tests:
python -m unittest discover

Ensure all tests pass to confirm that the setup was successful.

## Troubleshooting
If you encounter any issues:
- Check that all prerequisites are properly installed.
- Verify that the `.env` file contains correct API keys.
- Make sure that all dependencies are installed via `requirements.txt`.


## License
This project is licensed under the MIT License. See the LICENSE file for more details.