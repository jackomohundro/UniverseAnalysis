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
    if ticker.startswith('$'):
        return Cash()
    else:
        return client.get_last_trade(ticker=ticker)


def get_fund_data(df: pd.DataFrame, fund_name: str) -> pd.DataFrame:
    col_ticker = fund_name
    col_weight = f"{fund_name} Weight"

    if col_ticker not in df.columns or col_weight not in df.columns:
        raise ValueError(
            f"Could not find columns '{col_ticker}' and/or '{col_weight}' in the DataFrame."
        )

    # Subset to just these two columns, drop all-NaN rows
    sub_df = df[[col_ticker, col_weight]].dropna(how="all")

    # Rename to match your original logic
    sub_df.columns = ["Ticker", "Weight"]
    return sub_df


# **********************
# Load the active-fund and benchmark data
# **********************
portfolio_data_raw = pd.read_csv("Baskets/MUTF.csv")      # Active fund in a wide CSV
benchmark_data_raw = pd.read_csv("Baskets/BENCHMARK.csv") # Benchmark in a wide CSV


def determine_active_share(active_NAME, benchmark_NAME):
    """
    Merges the two wide CSV DataFrames on columns named:
      active_NAME  vs. benchmark_NAME
      active_NAME + ' Weight'  vs. benchmark_NAME + ' Weight'
    Then calculates Active Share.

    NOTE: This only works if your CSVs both contain columns
    named exactly active_NAME, active_NAME + ' Weight', etc.
    """
    df = pd.merge(
        portfolio_data_raw,
        benchmark_data_raw,
        left_on=active_NAME,
        right_on=benchmark_NAME,
        how="outer",
        suffixes=('_portfolio', '_benchmark')
    )

    df[f"{active_NAME} Weight"] = df[f"{active_NAME} Weight"].fillna(0)
    df[f"{benchmark_NAME} Weight"] = df[f"{benchmark_NAME} Weight"].fillna(0)

    df["Active_Difference"] = abs(
        df[f"{active_NAME} Weight"] - df[f"{benchmark_NAME} Weight"]
    )
    active_share = 0.5 * df["Active_Difference"].sum()

    print(f"Active Share: {active_share:.4f}")


def determine_sic_concentration(portfolio, weights):
    SICs_by_name = {}
    portfolio_SIC_weights = {}
    portfolio_SIC_weights[-1] = [0, "other", []]

    for i, ticker in enumerate(portfolio):
        print(ticker)
        try:
            response = r.get(
                f"https://api.polygon.io/v3/reference/tickers/{ticker.replace('.','')}?apiKey={API_KEY}"
            ).json()['results']

            sic_code = int(response["sic_code"])
            sic_desc = response['sic_description']
            SICs_by_name[ticker] = [sic_code, sic_desc]

            if sic_code not in portfolio_SIC_weights:
                portfolio_SIC_weights[sic_code] = [0, sic_desc, []]

            portfolio_SIC_weights[sic_code][0] += float(weights[i])
            portfolio_SIC_weights[sic_code][2].append(ticker)

        except Exception:
            SICs_by_name[ticker] = -1
            portfolio_SIC_weights[-1][0] += float(weights[i])

    return SICs_by_name, portfolio_SIC_weights


def determine_active_sector_weights(active, active_weights,
                                    benchmark, benchmark_weights,
                                    shortcut=True):
    active_portfolio_sectors = determine_sic_concentration(active, active_weights)[1]
    print("done with active portfolio")

    if shortcut:
        data = pd.read_csv("Baskets/SPY_sector_weights.csv")
        benchmark_portfolio_sectors = {
            int(row['SIC']): [row['Weight'], row['Name'], row['Securities']]
            for _, row in data.iterrows()
        }
    else:
        benchmark_portfolio_sectors = determine_sic_concentration(benchmark, benchmark_weights)[1]
    print("done with benchmark")

    active_sector_weights = {}
    all_sectors = set(active_portfolio_sectors.keys()).union(
        set(benchmark_portfolio_sectors.keys())
    )

    for sic_code in all_sectors:
        active_weight = active_portfolio_sectors.get(sic_code, [0, "", []])[0]
        benchmark_weight = benchmark_portfolio_sectors.get(sic_code, [0, "", []])[0]

        desc_active = active_portfolio_sectors.get(sic_code, ["", "unknown", []])[1]
        desc_bench  = benchmark_portfolio_sectors.get(sic_code, ["", "unknown", []])[1]
        description = desc_active if desc_active and desc_active != "unknown" else desc_bench

        active_sector_weights[sic_code] = {
            "sic_description": description,
            "active_weight": active_weight,
            "benchmark_weight": benchmark_weight,
            "difference": active_weight - benchmark_weight
        }

    df = pd.DataFrame.from_dict(active_sector_weights, orient='index')
    df.to_csv("active sector weights", index_label="SIC_Code")

    return active_sector_weights


def fetch_historical_prices(ticker, start_date, end_date):
    bars = client.get_aggs(ticker, 1, "day", start_date, end_date)
    prices = pd.DataFrame(bars, columns=["timestamp", "close"])
    prices["date"] = pd.to_datetime(prices["timestamp"], unit="ms").dt.date
    prices = prices.set_index("date")
    return prices[["close"]]


def calculate_returns(prices):
    data = prices.pct_change().dropna()
    for i, item in enumerate(data):
        # Very large spikes might be bad data
        if item > 2:
            data[i] = 0
    return data


def get_portfolio_returns(tickers, weights, start_date, end_date):
    if len(tickers) != len(weights):
        raise ValueError("Tickers and weights lists must have the same length.")

    weighted_returns = {}
    all_dates = None

    for ticker, weight in zip(tickers, weights):
        try:
            print(ticker)
            prices = fetch_historical_prices(ticker, start_date, end_date)
            prices = prices.dropna()
            prices = prices[prices["close"] > 0]

            returns = calculate_returns(prices["close"])

            if all_dates is None:
                all_dates = returns.index
            else:
                returns = returns.reindex(all_dates).fillna(0)

            weighted_returns[ticker] = returns * weight

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            continue

    portfolio_returns = pd.concat(weighted_returns, axis=1).fillna(0)
    portfolio_returns["Portfolio"] = portfolio_returns.sum(axis=1)
    return portfolio_returns["Portfolio"]


def get_total_return(ticker, start_date, end_date):
    """
    Used for sector-level return computations (Brinson).
    Returns a single total (cumulative) return from start_date to end_date.
    """
    try:
        prices = fetch_historical_prices(ticker, start_date, end_date)
        if prices.empty:
            return 0.0
        daily_returns = calculate_returns(prices["close"])
        return (1 + daily_returns).prod() - 1
    except Exception:
        return 0.0


def get_portfolio_sector_returns(portfolio, weights, start_date, end_date):
    """
    Group tickers by SIC code, compute Weighted Average Return within each SIC group.
    Return dict: {SIC_code: {"weight": float, "return": float, "description": str}}
    """
    _, sic_dict = determine_sic_concentration(portfolio, weights)
    sector_info = {}

    for sic_code, (sector_weight, sector_desc, sector_tickers) in sic_dict.items():
        if abs(sector_weight) < 1e-12:
            continue

        ticker_returns = []
        ticker_weights_within_sector = []

        for ticker in sector_tickers:
            w_portfolio = float(weights[portfolio.index(ticker)])
            fraction_of_sector = w_portfolio / sector_weight

            t_return = get_total_return(ticker, start_date, end_date)
            ticker_returns.append(t_return)
            ticker_weights_within_sector.append(fraction_of_sector)

        sector_return = sum(
            frac * ret for frac, ret in zip(ticker_weights_within_sector, ticker_returns)
        )

        sector_info[sic_code] = {
            "weight": sector_weight,
            "return": sector_return,
            "description": sector_desc,
        }

    return sector_info


def attribution(
    portfolio, portfolio_weights,
    benchmark, benchmark_weights,
    start_date, end_date
) :
    """
    Brinson-Fachler sector-level attribution:
      - Allocation
      - Selection
      - Interaction
      - Total
    """
    p_sector_data = get_portfolio_sector_returns(
        portfolio, portfolio_weights, start_date, end_date
    )
    b_sector_data = get_portfolio_sector_returns(
        benchmark, benchmark_weights, start_date, end_date
    )

    results = []
    all_sics = set(p_sector_data.keys()).union(b_sector_data.keys())

    for sic_code in all_sics:
        p_weight = p_sector_data.get(sic_code, {}).get("weight", 0.0)
        p_return = p_sector_data.get(sic_code, {}).get("return", 0.0)

        b_weight = b_sector_data.get(sic_code, {}).get("weight", 0.0)
        b_return = b_sector_data.get(sic_code, {}).get("return", 0.0)

        p_desc = p_sector_data.get(sic_code, {}).get("description", "")
        b_desc = b_sector_data.get(sic_code, {}).get("description", "")

        description = p_desc if p_desc else b_desc
        if not description:
            description = "N/A"

        alloc = (p_weight - b_weight) * b_return
        sel   = b_weight * (p_return - b_return)
        inter = (p_weight - b_weight) * (p_return - b_return)
        total = alloc + sel + inter

        results.append({
            "SIC_Code": sic_code,
            "Description": description,
            "Portfolio_Weight": p_weight,
            "Benchmark_Weight": b_weight,
            "Portfolio_Return": p_return,
            "Benchmark_Return": b_return,
            "Allocation": alloc,
            "Selection": sel,
            "Interaction": inter,
            "Total": total
        })

    df_attrib = pd.DataFrame(results)
    df_attrib.set_index("SIC_Code", inplace=True)

    allocation_sum  = df_attrib["Allocation"].sum()
    selection_sum   = df_attrib["Selection"].sum()
    interaction_sum = df_attrib["Interaction"].sum()
    total_sum       = df_attrib["Total"].sum()

    print("Attribution Summary:")
    print(f"Allocation Effect:  {allocation_sum:.4%}")
    print(f"Selection Effect:   {selection_sum:.4%}")
    print(f"Interaction Effect: {interaction_sum:.4%}")
    print(f"----------------------------")
    print(f"Total Difference:   {total_sum:.4%}")

    df_attrib.to_csv("attribution.csv")

    return df_attrib



