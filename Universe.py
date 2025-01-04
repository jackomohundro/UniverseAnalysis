import os
from dotenv import load_dotenv

import pandas as pd
from polygon import RESTClient
import requests as r

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

# **********************
# Load portfolio and benchmark data
portfolio_data = pd.read_csv("Baskets/MUTF.csv")
benchmark_data = pd.read_csv("Baskets/BENCHMARK.csv")
# **********************

def determine_active_share(active_NAME, benchmark_NAME) :

    df = pd.merge(portfolio_data, benchmark_data, left_on=active_NAME, right_on=benchmark_NAME, how="outer", suffixes=('_portfolio', '_benchmark'))

    df[f"{active_NAME} Weight"] = df[f"{active_NAME} Weight"].fillna(0)
    df[f"{benchmark_NAME} Weight"] = df[f"{benchmark_NAME} Weight"].fillna(0)

    df["Active_Difference"] = abs(df[f"{active_NAME} Weight"] - df[f"{benchmark_NAME} Weight"])

    active_share = 0.5 * df["Active_Difference"].sum()

    print(f"Active Share: {active_share:.4f}")

def determine_sic_concentration(portfolio, weights) :
    SICs_by_name = dict()
    portfolio_SIC_weights = dict()
    portfolio_SIC_weights[-1] = [0, "other",[]]
    for i, ticker in enumerate(portfolio) :
        print(ticker)
        try :
            response = r.get(f"https://api.polygon.io/v3/reference/tickers/{ticker.replace('.','')}?apiKey={API_KEY}").json()['results']
            SICs_by_name[ticker] = [response["sic_code"], response['sic_description']]

            if int(response['sic_code']) not in portfolio_SIC_weights :
                portfolio_SIC_weights[int(response['sic_code'])] = [0, response['sic_description'],[]]
            portfolio_SIC_weights[int(response['sic_code'])][0] += float(weights[i])
            portfolio_SIC_weights[int(response['sic_code'])][2].append(ticker)

        except :
            SICs_by_name[ticker] = -1
            portfolio_SIC_weights[-1][0] += weights[i]
    return SICs_by_name, portfolio_SIC_weights

def determine_active_sector_weights(active, active_weights, benchmark, benchmark_weights,shortcut=True) :
    active_portfolio_sectors = determine_sic_concentration(active, active_weights)[1]
    print("done with active portfolio")
    if (shortcut) :
        data = pd.read_csv("Baskets/SPY_sector_weights.csv")
        benchmark_portfolio_sectors = {
            int(row['SIC']): [row['Weight'], row['Name'], row['Securities']]
            for _, row in data.iterrows()
        }
    else :
        benchmark_portfolio_sectors = determine_sic_concentration(benchmark, benchmark_weights)[1]
    print("done with benchmark")

    active_sector_weights = {}

    all_sectors = set(active_portfolio_sectors.keys()).union(set(benchmark_portfolio_sectors.keys()))

    for sic_code in all_sectors:
        active_weight = active_portfolio_sectors.get(sic_code, [0])[0]
        benchmark_weight = benchmark_portfolio_sectors.get(sic_code, [0])[0]

        description = active_portfolio_sectors.get(sic_code, benchmark_portfolio_sectors.get(sic_code, ["", "unknown"]))[1]

        active_sector_weights[sic_code] = {
            "sic_description": description,
            "active_weight": active_weight,
            "benchmark_weight": benchmark_weight,
            "difference": active_weight - benchmark_weight
        }

    df = pd.DataFrame.from_dict(active_sector_weights, orient='index')

    # Save DataFrame to a CSV file
    df.to_csv("active sector weights", index_label="SIC_Code")

    return active_sector_weights

#data = determine_sic_concentration(list(portfolio_data['DAVP.X']),portfolio_data['DAVP.X Weight'])[1]
#print(data)
#pd.DataFrame(data).T.to_csv("sic.csv")

print(determine_active_sector_weights(list(portfolio_data['OAKM.X']), list(portfolio_data['OAKM.X Weight']),0,0))


