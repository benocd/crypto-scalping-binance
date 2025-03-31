import requests
import numpy as np
import time
import hmac
import hashlib
import json
import sys
import os
import csv
from datetime import datetime

# Binance API URLs
BASE_URL = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"
TICKER_ENDPOINT = "/api/v3/ticker/24hr"
ORDER_ENDPOINT = "/api/v3/order"

# Load API credentials from config.json
def load_api_keys():
    with open("config.json", "r") as file:
        config = json.load(file)
    return config["api_key"], config["api_secret"]

API_KEY, API_SECRET = load_api_keys()

BINANCE_FEE_RATE = 0.001  # 0.1% per trade, total 0.2% round-trip

# Trading configurations for different pairs
TRADING_CONFIG = {
    "BTCUSDT": {"trade_size": 0.0001, "sl_adjust": 100, "tp_adjust": 300},
    "ETHUSDT": {"trade_size": 0.005, "sl_adjust": 5, "tp_adjust": 15},
    "DOGEUSDT": {"trade_size": 50, "sl_adjust": 0.002, "tp_adjust": 0.005}
}

# Function to log trade details
def log_trade(symbol, side, quantity, timestamp):
    now = datetime.now()
    log_filename = f"logs/{now.year}_{now.month}.csv"

    # Check if file exists to write headers
    file_exists = os.path.isfile(log_filename)

    with open(log_filename, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write headers if file is new
        if not file_exists:
            writer.writerow(["Timestamp", "Symbol", "Side", "Quantity"])

        # Log trade details
        writer.writerow([datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d %H:%M:%S"), symbol, side, quantity])


# Fetch historical kline (candlestick) data
def get_klines(symbol, interval="5m", limit=50):
    url = f"{BASE_URL}{KLINES_ENDPOINT}?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    return response.json()

# Fetch 24hr ticker data
def get_ticker(symbol):
    url = f"{BASE_URL}{TICKER_ENDPOINT}?symbol={symbol}"
    response = requests.get(url)
    return response.json()

# Calculate moving averages
def moving_average(data, period):
    return np.convolve(data, np.ones(period)/period, mode='valid')[-1]

# Create a Binance order
def place_order(symbol, side, quantity, price=None, order_type="MARKET"):
    timestamp = int(time.time() * 1000)
    params = {
        "symbol": symbol,
        "side": side,
        "type": order_type,
        "quantity": quantity,
        "timestamp": timestamp
    }

    # If selling, account for the fee deduction
    if side == "SELL":
        quantity_after_fee = quantity * (1 - BINANCE_FEE_RATE)
        params["quantity"] = round(quantity_after_fee, 8)  # Binance typically accepts up to 8 decimals

    if order_type == "LIMIT":
        params["price"] = price
        params["timeInForce"] = "GTC"
    
    query_string = "&".join([f"{key}={value}" for key, value in params.items()])
    signature = hmac.new(API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    params["signature"] = signature
    headers = {"X-MBX-APIKEY": API_KEY}
    
    response = requests.post(f"{BASE_URL}{ORDER_ENDPOINT}", headers=headers, params=params)
    order_response = response.json()

    # Handle insufficient balance error
    if "code" in order_response and order_response["code"] == -2010:
        print("❌ ERROR: Insufficient balance. Stopping the script.")
        sys.exit(1)  # Exit program immediately
    else:
        try:
            log_trade(symbol, side, quantity, timestamp)
        except Exception as e:
            print(f"⚠️ WARNING: Failed to write log due to: {e}")

    return order_response

# Monitor price after trade execution
def monitor_trade(symbol, entry_price, stop_loss, take_profit, trade_size):
    while True:
        ticker = get_ticker(symbol)
        current_price = float(ticker["lastPrice"])
        
        print(f"📊 Monitoring {symbol}: Current Price {current_price:.6f}, TP {take_profit:.6f}, SL {stop_loss:.6f}")
        
        if current_price >= take_profit:
            print("🎯 Take Profit reached! Selling...")
            place_order(symbol, "SELL", trade_size)
            break
        elif current_price <= stop_loss:
            print("🛑 Stop Loss triggered! Selling...")
            place_order(symbol, "SELL", trade_size)
            break
        
        time.sleep(10)

# Perform analysis and execute trades for a given pair
def analyze_market(symbol, config):
    klines = get_klines(symbol)
    ticker = get_ticker(symbol)
    
    close_prices = np.array([float(candle[4]) for candle in klines])
    ma3 = moving_average(close_prices, 3)
    ma9 = moving_average(close_prices, 9)
    ma21 = moving_average(close_prices, 21)
    
    open_price = float(klines[-1][1])
    close_price = float(klines[-1][4])
    high_price = float(klines[-1][2])
    low_price = float(klines[-1][3])
    volume = float(ticker["volume"])
    
    trend = "Bearish" if close_price < open_price else "Bullish"
    momentum = "Strong" if abs(close_price - open_price) / open_price > 0.002 else "Weak"
    
    support = low_price if low_price > ma21 else ma21
    resistance = max(ma3, ma9)

    stop_loss = support - config["sl_adjust"]
    
    # Calculate adjusted TP to ensure it covers Binance fees
    min_tp_adjust = close_price * BINANCE_FEE_RATE * 2  # Minimum TP needed to cover fees
    final_tp_adjust = max(config["tp_adjust"], min_tp_adjust)  # Ensure TP is profitable

    take_profit = resistance + final_tp_adjust  # Use adjusted TP

    trade_plan = f"""
    📊 {symbol} Scalping Analysis:
    ----------------------------
    - Current Price: {close_price:.6f} USDT
    - Price Action: {trend} ({momentum} Momentum)
    - Moving Averages:
      - MA(3): {ma3:.6f}
      - MA(9): {ma9:.6f}
      - MA(21): {ma21:.6f} (Key Scalping Support)
    - Support Levels:
      - Immediate: {support:.6f}
    - Resistance Levels:
      - {resistance:.6f}
    
    Scalping Strategy:
    - Quick Entry: {ma3:.6f} (if price bounces above MA3)
    - Trend Confirmation: {ma9:.6f} (if price holds above MA9)
    - 🛑 Stop Loss: {stop_loss:.6f}
    - 🎯 Take Profit: {take_profit:.6f}
    """
    print(trade_plan)
    
    if close_price > ma3 and close_price > ma9:
        print(f"🚀 Placing Market Buy Order for {symbol}...")
        order_response = place_order(symbol, "BUY", config["trade_size"])
        print("Order Response:", json.dumps(order_response, indent=4))
        
        print(f"Setting TP at {take_profit:.6f} and SL at {stop_loss:.6f}")
        monitor_trade(symbol, close_price, stop_loss, take_profit, config["trade_size"])

# Main Loop: Iterate through all pairs
if __name__ == "__main__":
    while True:
        for pair, config in TRADING_CONFIG.items():
            analyze_market(pair, config)
            time.sleep(5)  # Wait before switching to the next pair
