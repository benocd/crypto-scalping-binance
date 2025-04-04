import requests
import numpy as np
import time
import hmac
import hashlib
import json
import sys
import os
import csv
from datetime import datetime, timezone
import math

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

MIN_SCORE_THRESHOLD = 9  # Adjust based on backtesting

# Trading configurations for different pairs
TRADING_CONFIG = {
    "BTCUSDT": {"trade_size": 0.0001, "sl_adjust": 100, "tp_adjust": 300, "quantity_precision": 8, "price_precision": 2},
    "ETHUSDT": {"trade_size": 0.005, "sl_adjust": 5, "tp_adjust": 15, "quantity_precision": 6, "price_precision": 2},
    "DOGEUSDT": {"trade_size": 50, "sl_adjust": 0.002, "tp_adjust": 0.005, "quantity_precision": 0, "price_precision": 6},
    "BNBUSDT": {"trade_size": 0.01, "sl_adjust": 2, "tp_adjust": 5, "quantity_precision": 3, "price_precision": 2},
    "SOLUSDT": {"trade_size": 0.1, "sl_adjust": 2, "tp_adjust": 5, "quantity_precision": 2, "price_precision": 2},
    "XRPUSDT": {"trade_size": 5, "sl_adjust": 0.005, "tp_adjust": 0.015, "quantity_precision": 0, "price_precision": 4},
    "LINKUSDT": {"trade_size": 0.5, "sl_adjust": 0.2, "tp_adjust": 0.5, "quantity_precision": 2, "price_precision": 2}
}

# Function to log trade details
def log_trade(order_json):

    # Extract relevant details
    transact_time = datetime.fromtimestamp(order_json["transactTime"] / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    order_id = order_json["orderId"]
    symbol = order_json["symbol"]
    order_type = order_json["type"]
    side = order_json["side"]

    # Extract fills (handling multiple fills)
    fills = order_json.get("fills", [])

    # Prepare data rows
    rows = [
        [
            transact_time,
            order_id,
            symbol,
            order_type,
            side,
            fill["price"],
            fill["qty"],
            fill["commission"],
            fill["commissionAsset"]
        ]
        for fill in fills
    ]

    now = datetime.now()
    log_filename = f"logs/{now.year}_{now.month}.csv"

    # Check if file exists to write headers
    file_exists = os.path.isfile(log_filename)

    with open(log_filename, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write headers if file is new
        if not file_exists:
            writer.writerow(["Timestamp", "Order ID", "Symbol", "Type", "Side", "Price", "Quantity", "Commission", "Commission Asset"])

        # Write trade rows
        writer.writerows(rows)

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

    # Get config settings for the asset
    config = TRADING_CONFIG.get(symbol, {"quantity_precision": 8, "price_precision": 8})
    quantity_precision = config["quantity_precision"]
    price_precision = config["price_precision"]

    params = {
        "symbol": symbol,
        "side": side,
        "type": order_type,
        "quantity": round(quantity, quantity_precision),
        "timestamp": timestamp
    }

    if order_type == "LIMIT":
        params["price"] = round(price,price_precision)
        params["timeInForce"] = "GTC"
    
    query_string = "&".join([f"{key}={value}" for key, value in params.items()])
    signature = hmac.new(API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    params["signature"] = signature
    headers = {"X-MBX-APIKEY": API_KEY}
    
    response = requests.post(f"{BASE_URL}{ORDER_ENDPOINT}", headers=headers, params=params)
    order_response = response.json()

    print(order_response) #Debug

    # Check if an error occurred
    if "code" in order_response:
        error_code = order_response["code"]
        error_msg = order_response["msg"]
        
        print(f"❌ Order failed! Error {error_code}: {error_msg}")
        
        # Handle specific errors
        if error_code == -2010:  # Insufficient balance
            print("⚠️ Insufficient balance. Reduce trade size or deposit more funds.")
        elif error_code == -1111:  # Precision issue
            print("⚠️ Precision error! Adjust quantity rounding for this asset.")
        elif error_code == -1013:  # Invalid quantity or price
            print("⚠️ Invalid quantity or price. Ensure it matches Binance requirements.")

        print(params)
        
        # You can raise an exception or return here to prevent further execution
        #raise Exception(f"Binance API Error {error_code}: {error_msg}")
        sys.exit(1)  # Exit program immediately
    else:
        try:
            log_trade(order_response)
        except Exception as e:
            print(f"⚠️ WARNING: Failed to write log due to: {e}")

    return order_response

# Monitors the trade and adjusts the stop-loss dynamically when the price moves up.
def monitor_trade(symbol, entry_price, stop_loss, take_profit, trade_size):
    highest_price = entry_price  # Track highest price reached
    start_time = datetime.now()
    extra_time = 0

    while True:
        try:
            ticker = get_ticker(symbol)
            current_price = float(ticker["lastPrice"])

            # Dynamically Adjust Stop Loss as Price Rises
            if current_price > highest_price:
                highest_price = current_price
                if(current_price > (entry_price * (1 + BINANCE_FEE_RATE * 5))):
                    #stop_loss = entry_price + 0.75 * (current_price - entry_price)
                    stop_loss = current_price - (current_price * BINANCE_FEE_RATE)
                elif(current_price > (entry_price * (1 + BINANCE_FEE_RATE * 4))):
                    stop_loss = (current_price + (entry_price * (1 + BINANCE_FEE_RATE * 2))) / 2
                else:
                    stop_loss = max(stop_loss, highest_price - TRADING_CONFIG[symbol]["sl_adjust"])
                print(f"🔄 SL Updated: {stop_loss:.6f} (Highest Price: {highest_price:.6f})")

            print(f"📊 Monitoring {symbol}: Current: {current_price:.6f}, TP: {take_profit:.6f}, SL: {stop_loss:.6f}")

            # Take Profit Hit → Place Limit Sell Order
            if current_price >= take_profit:
                print(f"🎯 Take Profit reached! Selling at {current_price:.6f}...")
                rounded_trade_size = round(trade_size * (1 - BINANCE_FEE_RATE), 8)
                order_response = place_order(symbol, "SELL", rounded_trade_size, price=take_profit, order_type="LIMIT")
                print("Sell Order Response:", json.dumps(order_response, indent=4))
                break

            # Stop Loss Hit → Market Sell
            elif current_price <= stop_loss:
                print("🛑 Stop Loss triggered! Selling...")
                rounded_trade_size = round(trade_size * (1 - BINANCE_FEE_RATE), 8)
                place_order(symbol, "SELL", rounded_trade_size)
                print("🕐 Cooling down... 2 min")
                time.sleep(120)
                break

            # 🚨 Exit after 30 minutes if TP or SL isn't hit
            if (datetime.now() - start_time).seconds > (1800 + extra_time):  # 30 minutes
                consolidation_percentage = abs(current_price - highest_price) / highest_price * 100
                if highest_price == current_price or consolidation_percentage <= 0.25:
                    extra_time = extra_time + 60  # Extend time by 60 seconds
                    print(f"Consolidation detected: Price is within {consolidation_percentage:.2f}% of the highest price. Extra time added.")
                else:
                    print("⏳ Trade timeout! Exiting...")
                    rounded_trade_size = round(trade_size * (1 - BINANCE_FEE_RATE), 8)
                    place_order(symbol, "SELL", rounded_trade_size)  # Market sell to exit
                    break
        
        except Exception as e:
            print(f"⚠️ WARNING: {e}")
        
        time.sleep(10)

def get_fear_greed_index():
    """Fetches the latest Fear & Greed Index"""
    url = "https://api.alternative.me/fng/"
    response = requests.get(url)
    data = response.json()
    return int(data["data"][0]["value"])  # Convert value to integer

def calculate_trade_score(symbol, close_price, ma3, ma9, ma21, momentum, volume, atr, high_price, low_price):
    score = 0  
    
    # 📈 **Trend Confirmation**
    if close_price > ma3: score += 1  # Short-term trend is bullish
    if close_price > ma9: score += 2  # Mid-term trend is bullish
    if close_price > ma21: score += 3 # Long-term trend is bullish

    # 🚀 **Pumping Detection**
    if is_pumping(symbol):
        score += 5  # Strong bullish movement detected 

    # 📊 **Momentum Check**
    if momentum > 0.003:  # Adjust threshold based on backtesting
        score += 2  # Strong price movement

    # 💰 **Liquidity Check**
    if volume > 2_000_000:  # Filter low-volume assets
        score += 2  
    elif volume > 1_000_000:
        score += 1  

    # 📉 **Volatility Check (Avoid too-stable markets)**
    if atr < (close_price * 0.002):  # ATR is less than 0.2% of price
        score -= 2  # Avoid low-volatility markets

    # 🎯 **Risk-Reward Filter**
    stop_loss_distance = abs(ma21 - close_price)
    take_profit_distance = abs(ma3 - close_price)
    
    if take_profit_distance >= (2 * stop_loss_distance):  # TP must be at least 2x SL
        score += 3  # Higher reward potential

    # 🚩 **Top of the Candle (Possible Exhaustion)**
    if close_price >= (high_price - ((high_price - low_price) * 0.15)):
        score -= 1  # Penalize if price is too high within the candle

    # 📊 **Fear & Greed Score Adjustment**
    fng_index = get_fear_greed_index()
    
    if fng_index > 60:  # Market is Greedy (Risk of Overvaluation)
        score -= 2
    elif fng_index < 40:  # Market is Fearful (Good Buy Opportunity)
        score += 2

    return score

# Perform analysis and execute trades for a given pair
def analyze_market(symbol, config, evaluate_only=False):
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

    # **Momentum Calculation**
    momentum = abs(close_price - open_price) / open_price

    # **Volatility Calculation (ATR)**
    atr = np.mean([float(k[2]) - float(k[3]) for k in klines])  # Avg High-Low range

    # **Compute the Score**
    score = calculate_trade_score(symbol, close_price, ma3, ma9, ma21, momentum, volume, atr, high_price, low_price)

    # Print Analysis
    trade_plan = f"""
    📊 {symbol} Scalping Analysis:
    - Trend: {"Bullish" if close_price > ma3 else "Bearish"}
    - Momentum: {momentum:.6f}
    - Volume: {volume:.2f}
    - Volatility (ATR): {atr:.6f}
    - Score: {score}
    """
    print(trade_plan)
    
    if evaluate_only:
        return score  

    # **Trade Execution** (Only if it passes the threshold)
    if score >= MIN_SCORE_THRESHOLD:
        print(f"🚀 Placing Market Buy Order for {symbol}...")
        order_response = place_order(symbol, "BUY", config["trade_size"])
        print("Order Response:", json.dumps(order_response, indent=4))
        
        take_profit = ma3 + config["tp_adjust"]
        stop_loss = ma21 - config["sl_adjust"]

        print(f"Setting TP at {take_profit:.6f} and SL at {stop_loss:.6f}")
        monitor_trade(symbol, close_price, stop_loss, take_profit, config["trade_size"])

def is_pumping(symbol, window=5):
    """
    Detects if the price of a coin is pumping based on a percentage increase over the last `window` minutes.
    Returns True if pumping, False otherwise.
    """
    pump_threshold = 1.5  # 1.5% price increase in 1 minute
    volume_threshold = 500000  # Min 500k USDT traded in the last 24 hours

    ticker = get_ticker(symbol)
    klines = get_klines(symbol, interval="1m", limit=window + 1)  # Get recent candles
    # Extract recent price changes
    close_prices = [float(candle[4]) for candle in klines]
    last_price = close_prices[-1]
    prev_price = close_prices[-2]
    price_change = (last_price - prev_price) / prev_price * 100  # % Change

    # Check volume
    volume = float(ticker["volume"])

    if price_change >= pump_threshold and volume > volume_threshold:
        print(f"🚀 {symbol} is PUMPING! Price change: {price_change:.2f}%")
        return True
    else:
        return False

if __name__ == "__main__":
    while True:
        print("🔎 Scanning market for best trading opportunity...")

        best_symbol = None
        best_score = float("-inf")

        for symbol, config in TRADING_CONFIG.items():
            score = analyze_market(symbol, config, evaluate_only=True)  

            if score > best_score:
                best_score = score
                best_symbol = symbol

        # ✅ Trade only if it meets the score threshold
        if best_symbol and best_score >= MIN_SCORE_THRESHOLD:
            print(f"🚀 Best trade opportunity: {best_symbol} (Score: {best_score})")
            analyze_market(best_symbol, TRADING_CONFIG[best_symbol])
        else:
            print(f"⚠️ No valid trade. Best Score: {best_score} (Threshold: {MIN_SCORE_THRESHOLD})")

        print("🕒 Waiting before next scan...")
        time.sleep(10)
