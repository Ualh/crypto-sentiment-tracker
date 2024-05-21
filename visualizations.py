import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_error, root_mean_squared_error
import statsmodels.api as sm
import urllib.parse
import pmdarima as pm
from sklearn.model_selection import train_test_split



class Visualizations:
    def __init__(self, window_size=7):
        """
        Initializes the Visualizations class with default settings for plotting.

        Parameters:
        - window_size (int): Window size for smoothing sentiment data.
        """
        self.window_size = window_size

    def average_sentiment_per_time(self, from_date, data):
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
            date_limit = pd.Timestamp.today() - pd.Timedelta(days=17.5)
        elif from_date in [30, 2]:
            data['date'] = data['date'].dt.round('h')
            # Get data from the last 30 days
            date_limit = pd.Timestamp.today() - pd.Timedelta(days=47.1)
        else:
            raise ValueError("from_date should be either 1 (daily) or 30 (monthly).")

        # Filter data to include only the desired date range
        filtered_data = data[data['date'] >= date_limit]

        average_sentiment_per_time = filtered_data.groupby('date', as_index=False)['sentiment'].mean()
        average_sentiment_per_time.columns = ['date', 'average sentiment']
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
        
        # Normalize the price data within each coin
        price_data['normalized price'] = price_data.groupby('coin name')['price'].transform(lambda x: MinMaxScaler().fit_transform(x.values.reshape(-1, 1)).flatten())
    
        # Aggregate normalized prices by averaging them across all coins for each unique timestamp
        aggregated_data = price_data.groupby('timestamp', as_index=False)['normalized price'].mean()
        aggregated_data.columns = ['timestamp', 'normalized price']
        
        return aggregated_data
    
   

    def plot_normalized_price_and_sentiment(self, price_data, sentiment_data, future_predictions=None, for_web=False):
        """
        Plots normalized price and sentiment data over time, including future predictions if provided.

        Parameters:
        - price_data (DataFrame): DataFrame containing 'Timestamp' and 'Normalized Price' columns.
        - sentiment_data (DataFrame): DataFrame containing 'Date' and 'Average Sentiment' columns.
        - future_predictions (list): List of predicted future prices.
        """
        price_data.set_index('timestamp', inplace=True)
        sentiment_data.set_index('date', inplace=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, 10))
        ax.plot(price_data.index, price_data['normalized price'], label='Actual Normalized Price', color=colors[0])
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Price (0 to 1 Scale)', color=colors[0])
        ax.set_title('Price Trend with Sentiment Analysis')
        ax.tick_params(axis='y', labelcolor=colors[0])

        if future_predictions:
            future_dates = pd.date_range(start=price_data.index[-1], periods=len(future_predictions) + 1, freq='D')[1:]
            ax.plot(future_dates, future_predictions, label='Predicted Price', color='red')

        ax2 = ax.twinx()
        ax2.plot(sentiment_data.index, sentiment_data['average sentiment'].rolling(window=self.window_size).mean(), label='Smoothed Sentiment', color=colors[5])
        ax2.set_ylabel('Sentiment (0 to 1 Scale)', color=colors[5])
        ax2.tick_params(axis='y', labelcolor=colors[5])
        plt.xticks(rotation=45)

        # Add legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)

        plt.grid(True)

        if for_web:
            buf = BytesIO()
            plt.savefig(buf, format='svg', bbox_inches='tight')
            buf.seek(0)
            svg = buf.getvalue().decode('utf-8')
            buf.close()
            plt.close(fig)
            return svg
        else:
            plt.show()
            return fig

        

    def analysis(self, combined_data, from_date, model_type='linear', for_web=False, predict_days=7):
        if from_date == 1:
            time = '5min'
            period = 'Per 5 Minute'
            intervals = '30min'
            lags = [-10, -5, -1, 0, 1, 5, 10]
        elif from_date in [30, 2]:
            time = 'Day'
            period = 'Daily'
            intervals = 'W'
            lags = [-7, -2, -1, 0, 1, 2, 7]

        colors = plt.cm.viridis(np.linspace(0, 1, 10))
        correlations = []
        results = []
        future_predictions_by_lag = []

        for lag in lags:
            temp_data = combined_data.copy()
            temp_data['Lagged Sentiment'] = temp_data['average sentiment'].shift(lag)
            temp_data.dropna(inplace=True)

            if temp_data.empty or len(temp_data) < 2:
                print(f"Not enough data points to perform fit for lag {lag}")
                continue

            correlation = temp_data[['Lagged Sentiment', f'Next {time} Price Change']].corr().iloc[0, 1]
            correlations.append((lag, correlation))

            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(temp_data[['Lagged Sentiment']], temp_data[f'Next {time} Price Change'], test_size=0.25, random_state=42)

            if model_type == 'linear':
                X_train_const = sm.add_constant(X_train)
                X_test_const = sm.add_constant(X_test)
                model = sm.OLS(y_train, X_train_const).fit()
                y_pred = model.predict(X_test_const)
                rsquared = r2_score(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                # Predict future price changes
                future_sentiment = combined_data['average sentiment'].tail(predict_days).shift(lag).fillna(method='ffill')
                future_pred = model.predict(sm.add_constant(future_sentiment))
            else:
                rf_model = RandomForestRegressor(n_estimators=1000, random_state=42)
                rf_model.fit(X_train, y_train)
                y_pred = rf_model.predict(X_test)
                rsquared = r2_score(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                # Predict future price changes
                future_sentiment = combined_data['average sentiment'].tail(predict_days).shift(lag).fillna(method='ffill')
                future_pred = rf_model.predict(future_sentiment.values.reshape(-1, 1))


            future_predictions_by_lag.append(future_pred.mean(axis=0))

            if for_web:
                fig, ax = plt.subplots(figsize=(8, 6))
                scatter = ax.scatter(temp_data['Lagged Sentiment'], temp_data[f'Next {time} Price Change'], 
                                    c=temp_data[f'Next {time} Price Change'], cmap='viridis')
                plt.title(f'{period} Sentiment vs. Lag {lag} Price Change')
                plt.xlabel('Lagged Sentiment')
                plt.ylabel(f'Next {time} Price Change')
                plt.grid(True)
                m, b = np.polyfit(temp_data['Lagged Sentiment'], temp_data[f'Next {time} Price Change'], 1)
                plt.plot(temp_data['Lagged Sentiment'], m * temp_data['Lagged Sentiment'] + b, color='darkred')
                img = BytesIO()
                plt.savefig(img, format='svg', bbox_inches='tight')
                img.seek(0)
                svg_data = img.getvalue().decode('utf-8')
                svg_url = "data:image/svg+xml;utf8," + urllib.parse.quote(svg_data)
                plt.close()

                stats = {
                    "correlation": round(correlation, 2),
                    "average_price_change_high": round(temp_data[temp_data['average sentiment'] > temp_data['average sentiment'].median()][f'Next {time} Price Change'].mean(), 2),
                    "average_price_change_low": round(temp_data[temp_data['average sentiment'] <= temp_data['average sentiment'].median()][f'Next {time} Price Change'].mean(), 2),
                    "rsquared": round(rsquared, 2),
                    "rmse": round(rmse, 2)
                }

                results.append({"plot_url": svg_url, "stats": stats, "lag": lag})
            else:
                plt.scatter(temp_data['Lagged Sentiment'], temp_data[f'Next {time} Price Change'], 
                            c=temp_data[f'Next {time} Price Change'], cmap='viridis')
                plt.title(f'{period} Sentiment vs. Lag {lag} Price Change')
                plt.xlabel('Lagged Sentiment')
                plt.ylabel(f'Next {time} Price Change')
                plt.grid(True)
                m, b = np.polyfit(temp_data['Lagged Sentiment'], temp_data[f'Next {time} Price Change'], 1)
                plt.plot(temp_data['Lagged Sentiment'], m * temp_data['Lagged Sentiment'] + b, color='darkred')
                plt.show()

                print(f"Correlation with {lag} {time}(s) lag: {round(correlation, 2)}")
                print(f"{model_type.capitalize()} Model R-squared: {round(rsquared, 2)}")
                print(f"{model_type.capitalize()} Model RMSE: {round(rmse, 2)}")

                median_sentiment = temp_data['average sentiment'].median()
                high_sentiment = temp_data[temp_data['average sentiment'] > median_sentiment]
                low_sentiment = temp_data[temp_data['average sentiment'] <= median_sentiment]
                print(f"\nAverage Price Change on High Sentiment {time}s:", round((high_sentiment[f'Next {time} Price Change'].mean()) * 100), "%")
                print(f"Average Price Change on Low Sentiment {time}s:", round((low_sentiment[f'Next {time} Price Change'].mean()) * 100), "%")

        if not for_web:
            correlation_summary = pd.DataFrame(correlations, columns=[f'Lag ({time})', 'Correlation'])
            plt.figure(figsize=(8, 4))
            sns.barplot(x=f'Lag ({time})', y='Correlation', data=correlation_summary, color=colors[0])
            plt.title('Correlation of Sentiment and Price Change Over Different Lags')
            plt.ylabel('Correlation Coefficient')
            plt.xlabel(f'{time}s of Lag')
            plt.show()

        if for_web:
            return results, future_predictions_by_lag
        else:
            return future_predictions_by_lag

    def forecast_prices_with_arima(self, price_data, forecast_periods=24, for_web=False):
        """
        Forecasts average percentage change in future prices using an Auto ARIMA model on a copy of aggregated and normalized price data,
        and optionally plots the forecast with the historical data.

        Parameters:
        - price_data (DataFrame): DataFrame containing 'timestamp' and 'normalized price' columns, aggregated across different coins.
        - forecast_periods (int): Number of periods to forecast ahead.
        - for_web (bool): If True, returns the plot in SVG format, otherwise displays the plot interactively.

        Returns:
        - tuple (dict, str/svg): Returns a tuple containing the averages of forecasted changes, RMSE, and the plot either as a path to save or as an SVG string.
        """
        data_copy = price_data.copy()
        data_copy.columns = data_copy.columns.str.lower()
        last_known_price = data_copy['normalized price'].iloc[-1]

        # Fit the Auto ARIMA model
        model = pm.auto_arima(data_copy['normalized price'], seasonal=False, m=1,
                            suppress_warnings=True, stepwise=True, error_action='ignore', trace=False)

        # In-sample prediction for RMSE calculation
        in_sample_pred = model.predict_in_sample()
        rmse = np.sqrt(mean_squared_error(data_copy['normalized price'], in_sample_pred))

        # Forecast future prices
        forecast_results = model.predict(n_periods=forecast_periods, return_conf_int=True)
        forecast_df = pd.DataFrame({
            'forecasted_price': forecast_results[0],
            'ci_lower': forecast_results[1][:, 0],
            'ci_upper': forecast_results[1][:, 1]
        })

        # Add a timestamp index to the forecast DataFrame
        last_date = data_copy.index[-1]
        future_dates = pd.date_range(start=last_date, periods=forecast_periods + 1, freq='h')[1:]
        forecast_df.set_index(future_dates, inplace=True)
        
        # Calculate the average percentage change and the confidence intervals in percentage
        forecast_df['avg_change_percent'] = ((forecast_df['forecasted_price'] - last_known_price) / last_known_price) * 100
        forecast_df['ci_lower_percent'] = ((forecast_df['ci_lower'] - last_known_price) / last_known_price) * 100
        forecast_df['ci_upper_percent'] = ((forecast_df['ci_upper'] - last_known_price) / last_known_price) * 100

        # Calculate the average of the entire forecast period for each measure
        averages = {
            'average_change_percent': forecast_df['avg_change_percent'].mean(),
            'average_ci_lower_percent': forecast_df['ci_lower_percent'].mean(),
            'average_ci_upper_percent': forecast_df['ci_upper_percent'].mean(),
            'rmse': rmse
        }

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(data_copy.index, data_copy['normalized price'], label='Historical Normalized Price')
        plt.plot(forecast_df.index, forecast_df['forecasted_price'], label='Forecasted Price', color='red')
        plt.fill_between(forecast_df.index, forecast_df['ci_lower'], forecast_df['ci_upper'], color='red', alpha=0.3)
        plt.title('Forecasted Normalized Prices')
        plt.xlabel('Timestamp')
        plt.ylabel('Normalized Price')
        plt.legend()

        if for_web:
            buf = BytesIO()
            plt.savefig(buf, format='svg', bbox_inches='tight')
            plt.close()
            buf.seek(0)
            svg = buf.getvalue().decode('utf-8')
            return averages, svg
        else:
            plt.show()
            return averages, None