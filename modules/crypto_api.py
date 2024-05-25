import requests
import csv

class CryptoAPI:
    def __init__(self, base_url="https://api.alternative.me/v2"):
        self.base_url = base_url

    def get_crypto_listings(self):
        """Fetches an overview of all available cryptocurrencies."""
        endpoint = "/listings/"
        url = self.base_url + endpoint
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to retrieve data: HTTP {response.status_code}")
            return None

    def save_crypto_listings_to_csv(self):
        """Saves the list of available cryptocurrencies to a CSV file."""
        listings = self.get_crypto_listings()
        if listings:
            num_cryptocurrencies = listings.get("metadata", {}).get("num_cryptocurrencies", 0)
            print(f"Number of cryptocurrencies: {num_cryptocurrencies}")
            
            with open("data/crypto_listings.csv", "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["name", "symbol"])  # Write header row
                
                for crypto in listings.get("data", []):
                    name = crypto.get("name", "")
                    symbol = crypto.get("symbol", "")
                    writer.writerow([name, symbol])
                                        
            print("CSV file saved successfully")
        else:
            print("Failed to retrieve data, CSV file not saved")