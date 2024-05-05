import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm

class Visualizations:
    def __init__(self, window_size=7):
        """
        Initializes the Visualizations class with default settings for plotting.

        Parameters:
        - window_size (int): Window size for smoothing sentiment data.
        """
        self.window_size = window_size

    def average_sentiment_per_time(self,from_date, data):
        """
        Averages sentiment data for each unique timestamp per hour.

        Parameters:
        - data (DataFrame): DataFrame containing 'date' and 'sentiment' columns.

        Returns:
        - DataFrame: DataFrame with the average sentiment calculated for each unique timestamp.
        """
        data.columns = data.columns.str.lower()
        data['date'] = pd.to_datetime(data['date'])

        # Determine the date range to filter data
        if from_date == 1:
            # Get data from the last 2 days
            date_limit = pd.Timestamp.today() - pd.Timedelta(days=1.2)
        elif from_date == 30:
            data['date'] = data['date'].dt.round('h')
            # Get data from the last 30 days
            date_limit = pd.Timestamp.today() - pd.Timedelta(days=31)
        else:
            raise ValueError("from_date should be either 1 (daily) or 30 (monthly).")
        
        # Filter data to include only the desired date range
        filtered_data = data[data['date'] >= date_limit]

        average_sentiment_per_time = filtered_data.groupby('date', as_index=False)['sentiment'].mean()
        average_sentiment_per_time.columns = ['Date', 'Average Sentiment']
        return average_sentiment_per_time

    def normalize_and_aggregate_prices(self, price_data):
        """
        Normalizes and aggregates price data by timestamp across different coins.

        Parameters:
        - price_data (DataFrame): DataFrame containing 'Coin Name', 'Timestamp', and 'Price' columns.

        Returns:
        - DataFrame: DataFrame containing normalized and aggregated price data.
        """
        price_data.columns = map(str.lower, price_data.columns)
        price_data['timestamp'] = pd.to_datetime(price_data['timestamp'], unit='s')
        
        # Normalize the price data across all coins at each timestamp
        scaler = MinMaxScaler()
        price_data['normalized price'] = scaler.fit_transform(price_data[['price']])
        
        # Aggregate normalized prices by averaging them across all coins for each unique timestamp
        aggregated_data = price_data.groupby('timestamp', as_index=False)['normalized price'].mean()
        aggregated_data.columns = ['timestamp', 'normalized price']
        
        return aggregated_data

    def plot_normalized_price_and_sentiment(self, price_data, sentiment_data):
        """
        Plots normalized price and sentiment data over time.

        Parameters:
        - price_data (DataFrame): DataFrame containing 'Timestamp' and 'Normalized Price' columns.
        - sentiment_data (DataFrame): DataFrame containing 'Date' and 'Average Sentiment' columns.
        """
        price_data.set_index('timestamp', inplace=True)
        sentiment_data.set_index('Date', inplace=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, 10))
        ax.plot(price_data.index, price_data['normalized price'], label='Normalized Price', color=colors[0])
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Price (0 to 1 Scale)', color=colors[0])
        ax.set_title('Price Trend with Sentiment Analysis')
        ax.tick_params(axis='y', labelcolor=colors[0])
        ax.legend()

        ax2 = ax.twinx()
        ax2.plot(sentiment_data.index, sentiment_data['Average Sentiment'].rolling(window=self.window_size).mean(), label='Smoothed Sentiment', color=colors[5], linestyle='--')
        ax2.set_ylabel('Sentiment (0 to 1 Scale)', color=colors[5])
        ax2.legend(loc='upper left')
        ax2.tick_params(axis='y', labelcolor=colors[5])

        fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
        plt.grid(True)
        plt.show()
        return fig

    def analysis(self, combined_data, from_date):
        if from_date == 1:
            time = '5min'
            period = 'Per 5 Minute'
            intervals = '30min'
            lags = [-10, -5, -1, 0, 1, 5, 10]
        elif from_date == 30:
            time = 'Day'
            period = 'Daily'
            intervals = 'W'
            lags = [-7, -2, -1, 0, 1, 2, 7]
        colors = plt.cm.viridis(np.linspace(0, 1, 10))
        correlations = []
        for lag in lags:
            temp_data = combined_data.copy()
            temp_data['Lagged Sentiment'] = temp_data['Average Sentiment'].shift(lag)
            temp_data.dropna(inplace=True)
            correlation = temp_data[['Lagged Sentiment', f'Next {time} Price Change']].corr().iloc[0, 1]
            correlations.append((lag, correlation))
            plt.scatter(temp_data['Lagged Sentiment'], temp_data[f'Next {time} Price Change'], c=temp_data[f'Next {time} Price Change'], cmap='viridis')
            plt.title(f'{period} Sentiment vs. Lag {lag} Price Change')
            plt.xlabel('Lagged Sentiment')
            plt.ylabel(f'Next {time} Price Change')
            plt.grid(True)
            m, b = np.polyfit(temp_data['Lagged Sentiment'], temp_data[f'Next {time} Price Change'], 1)
            plt.plot(temp_data['Lagged Sentiment'], m * temp_data['Lagged Sentiment'] + b, color='darkred')
            plt.show()

            print(f"Correlation with {lag} {time}(s) lag: {round(correlation,2)}")

            median_sentiment = temp_data['Average Sentiment'].median()
            high_sentiment = temp_data[temp_data['Average Sentiment'] > median_sentiment]
            low_sentiment = temp_data[temp_data['Average Sentiment'] <= median_sentiment]
            print(f"\nAverage Price Change on High Sentiment {time}s:", round((high_sentiment[f'Next {time} Price Change'].mean())*100), "%")
            print(f"Average Price Change on Low Sentiment {time}s:", round((low_sentiment[f'Next {time} Price Change'].mean())*100), "%")

            X = sm.add_constant(temp_data['Lagged Sentiment'])
            model = sm.OLS(temp_data[f'Next {time} Price Change'], X).fit()
            rsquared = model.rsquared
            aic = model.aic
            pvalue = model.pvalues['Lagged Sentiment']
            print("\nPredictive Power of Sentiment on Price:")
            print("R-squared:", round(rsquared,2))
            print("AIC:", round(aic,2))
            print("P-value of Lagged Sentiment variable:", round(pvalue,2))
            
            if from_date == 30:
                resampled_data = temp_data.resample(intervals).mean()
                X = sm.add_constant(resampled_data['Lagged Sentiment'])
                model = sm.OLS(resampled_data[f'Next {time} Price Change'], X).fit()
                rsquared = model.rsquared
                aic = model.aic
                pvalue = model.pvalues['Lagged Sentiment']
                print(f"\nPredictive Power of Sentiment on Price, over a {intervals} period:")
                print("R-squared:", round(rsquared,2))
                print("AIC:", round(aic,2))
                print("P-value of Lagged Sentiment variable:", round(pvalue,2))

        correlation_summary = pd.DataFrame(correlations, columns=[f'Lag ({time})', 'Correlation'])
        plt.figure(figsize=(8, 4))
        sns.barplot(x=f'Lag ({time})', y='Correlation', data=correlation_summary, color=colors[0])
        plt.title('Correlation of Sentiment and Price Change Over Different Lags')
        plt.ylabel('Correlation Coefficient')
        plt.xlabel(f'{time}s of Lag')
        plt.show()
