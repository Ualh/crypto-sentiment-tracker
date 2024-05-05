import pandas as pd
from datetime import datetime, timedelta

class DataHandler:
    def __init__(self):
        # Initialization can be used for setting global properties or configurations if needed
        pass

    def process_duplicates(self, df):
        """
        processes the text fields for duplicates

        Args:
        - df (dataframe): The dataframe you want to process duplicates for

        Returns:
        - None
        """
        # Normalize text fields to prevent case sensitivity or trailing spaces from causing issues
        df['headline'] = df['headline'].str.strip().str.lower()
        # Create a new column for partial headline (first 3 words)
        df['partial_headline'] = df['headline'].apply(lambda x: ' '.join(x.split()[:10]))
        df.drop_duplicates(subset=['partial_headline'], inplace=True)
        print("Duplicate Entries:", df.duplicated(subset=['partial_headline']).sum())
        df.drop('partial_headline', axis=1, inplace=True)
        

    def get_date_str(self, days):
        return (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')

    def get_date_dt(self, days):
        return datetime.strptime((datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d'), '%Y-%m-%d')

    def standardize_date_format(self, df, date_column='date'):
        df[date_column] = pd.to_datetime(df[date_column]).dt.strftime('%Y-%m-%d %H:%M:%S')
        return df

    def convert_to_unix_ms(self, dt):
        return int(dt.timestamp() * 1000)

    def convert_to_unix_seconds(self, dt):
        return int(dt.timestamp())

    def get_start_today_date(self):
        return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    def get_end_today_date(self):
        return self.get_start_today_date() + timedelta(days=1) - timedelta(seconds=1)
