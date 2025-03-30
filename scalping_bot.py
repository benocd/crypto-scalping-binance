import requests
import numpy as np
import time
import hmac
import hashlib
import json

# Binance API URLs
BASE_URL = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"
TICKER_ENDPOINT = "/api/v3/ticker/24hr"
ORDER_ENDPOINT = "/api/v3/order"

# Binance API keys (replace with your own keys)
API_KEY = "your_api_key"
API_SECRET = "your_api_secret"

# Trading configurations for different pairs
TRADING_CONFIG = {
    "DOGEUSDT": {"trade_size": 50, "sl_adjust": 0.002, "tp_adjust": 0.005},
    "ETHUSDT": {"trade_size": 0.01, "sl_adjust": 5, "tp_adjust": 15},
    "BTCUSDT": {"trade_size": 0.001, "sl_adjust": 100, "tp_adjust": 300}
}

# Define trading pair
TRADING_PAIR = "DOGEUSDT"  # Change this as needed
CONFIG = TRADING_CONFIG[TRADING_PAIR]

# Fetch historical kline (candlestick) data
def get_klines(symbol=TRADING_PAIR, interval="5m", limit=50):
    url = f"{BASE_URL}{KLINES_ENDPOINT}?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    return response.json()

# Fetch 24hr ticker data
def get_ticker(symbol=TRADING_PAIR):
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
    if order_type == "LIMIT":
        params["price"] = price
        params["timeInForce"] = "GTC"
    
    query_string = "&".join([f"{key}={value}" for key, value in params.items()])
    signature = hmac.new(API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    params["signature"] = signature
    headers = {"X-MBX-APIKEY": API_KEY}
    
    response = requests.post(f"{BASE_URL}{ORDER_ENDPOINT}", headers=headers, params=params)
    return response.json()

# Monitor price after trade execution
def monitor_trade(entry_price, stop_loss, take_profit):
    while True:
        ticker = get_ticker()
        current_price = float(ticker["lastPrice"])
        
        print(f"Monitoring trade: Current Price {current_price:.6f}, TP {take_profit:.6f}, SL {stop_loss:.6f}")
        
        if current_price >= take_profit:
            print("Take Profit reached! Selling...")
            place_order(TRADING_PAIR, "SELL", CONFIG["trade_size"])
            break
        elif current_price <= stop_loss:
            print("Stop Loss triggered! Selling...")
            place_order(TRADING_PAIR, "SELL", CONFIG["trade_size"])
            break
        
        time.sleep(10)

# Perform analysis and execute trades
def analyze_market():
    klines = get_klines()
    ticker = get_ticker()
    
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
    
    stop_loss = support - CONFIG["sl_adjust"]
    take_profit = resistance + CONFIG["tp_adjust"]
    
    trade_plan = f"""
    {TRADING_PAIR} Scalping Analysis:
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
    - Stop Loss: {stop_loss:.6f}
    - Take Profit: {take_profit:.6f}
    """
    print(trade_plan)
    
    if close_price > ma3 and close_price > ma9:
        print("Placing Market Buy Order...")
        order_response = place_order(TRADING_PAIR, "BUY", CONFIG["trade_size"])
        print("Order Response:", json.dumps(order_response, indent=4))
        
        print(f"Setting Take Profit at {take_profit:.6f} and Stop Loss at {stop_loss:.6f}")
        monitor_trade(close_price, stop_loss, take_profit)
    
if __name__ == "__main__":
    analyze_market()
