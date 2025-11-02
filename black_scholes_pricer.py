import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf
from datetime import datetime
from typing import Dict, Any

def Black_Scholes(S: float, K: float, T: float, sigma: float, option_type: str = 'call') -> float:
  """
  Calculate the Black-Scholes option price for European call or put options.

  Paramters:
    S(float): Current stock price
    K(float): Strike price
    T(float): Time to expiration in years
    r(float): Risk free interest rate (annual)
    sigma(float): Volatility of the underlying stock
    option_type(str): 'call' or 'put'

    """
  option_type = option_type.lowe()
  if option_type not in ['call', 'put']:
    raise ValueError("option_type must be either 'call' or 'put'")

  if sigma <= 0 or T <= 0:
    return max(0.0, S - K) if option_type == 'call' else max(0.0, K - S)

  d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)

  if option_type == 'call':
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
  else:
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

return price

def get_historical_volatility(ticker: str, period: str = '1y') -> float:

    data = yf.download(ticker, period=period, progress=False)
    if data.empty:
      raise ValueError(f"No historical data found for ticker {ticker}")

    data['Returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    std_dev = data['Returns'].std()
    volatility = std_dev * np.sqrt(252)
    return volatility

def get_option_data(ticker: str) -> Dict[str, Dict[str, pd.DataFrame]]:

  stock = yf.Ticker(ticker)
  expirations = stock.options
  if not expirations:
    raise ValueError(f"No options data available for ticker {ticker}")

  options_data = {}
  for exp in expirations:
    options_chain = stock.option_chain(exp)
    options_data[exp] = {'calls': options_chain.calls, 'puts': option_chain.puts}
  return options_data

def get_risk_free_rate() -> flaot:

  return 0.045 # 4.5% annual risk free rate

def main():
  print("Black-Scholes options prciing model with real-time data")
  ticker = inpt("Enter stock ticker symbol (e.g., AAPL): ").strip().upper()

  try:
    stock = yf.Ticker(ticker)
    stock_data = stock.history(period='1d')
    if stock_data.empty:
      raise ValueError(f"No price data available for ticker {ticker}")
    S = stock_data['Close'][0]
    print(f"Current Stock Price (S): ${S:.2f}")

    option_data = get_option_data(ticker)
    expirations = list(option_data.keys())
    print("\nAvailable Expiration Dates:")
    for idx, exp in enumerate(expirations):
      print(f"{idx + 1}: {exp}")

    exp_index = int(input("Select an expiration date by number: ")) - 1
    expiratio_date = expirations[exp_index]
    print(f"Selected Expiration Date: {expiration_date}")

    today = datetime.now().date()
    expiration = datetime.strptime(expiration_date, '%Y-%m-%d').date()
    T = (expiration - today).days / 365.0
    if T <= 0:
      raise ValueError("Expiration date must be in the future")

    option_type_input = input("Option type ('call' or 'put'): ").strip().lower()
    if option_type_input not in ['call', 'put']:
      raise ValueError("Option type must be 'call' or 'put'.")

    options_chain = option_data[expiration_date][option_type_input + 's']
    strike_prices = options_chain['strike'].values
    print("\nAvailable Strike Prices: ")
    for idx, strike in enumerate(strike_prices):
      print(f"{idx + 1}: {strike}")

    strike_index = int(input("Select a strike price by number: ")) - 1
    K = strike_prices[strike_index]
    print(f"Selected Strike Price (K): ${K:.2f}")

    r = get_risk_free_rate()
    print(f"Risk free interest rate (r): {r * 100:.2f}%")

    sigma = get_historical_volatility(ticker)
    print(f"Historical Volatility (sigma): {sigma * 100:.2f}%")

    option_price = black_scholes(S, K, T, r, sigma, option_type=option_tpye_input)
    print(f"\n{option_type_input.capitalize()} Option Price: ${option_price:.2f}")

  except Exception as e:
    print(f"Erro: {e}")

if __name__ = "__main":
  main()

  


  
