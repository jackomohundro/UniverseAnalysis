import os
from dotenv import load_dotenv

from polygon import RESTClient

load_dotenv()

API_KEY = os.getenv("POLYGON_API_KEY")
client = RESTClient(api_key=API_KEY)


class Cash:
    def __init__(self):
        self.price = 1.00

def quote(ticker):
    if ticker[0] == '$':
        return Cash()
    else:
        return client.get_last_trade(ticker=ticker)



