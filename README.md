# Binance Scalping Trading Bot

## Overview
This Python script automates scalping trades for cryptocurrency pairs on Binance, leveraging moving average (MA) strategies. It analyzes market conditions, executes buy orders based on trend confirmation, and monitors trades to ensure optimal take-profit (TP) and stop-loss (SL) execution.

## Features
- Fetches real-time price data from Binance API.
- Calculates moving averages (MA3, MA9, MA21) to determine trade signals.
- Implements a scalping strategy with dynamic TP and SL levels.
- Automatically places and monitors trades.
- Supports multiple trading pairs with configurable settings.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/benocd/binance-scalping-bot.git
   cd binance-scalping-bot
   ```
2. Install required dependencies:
   ```bash
   pip install numpy requests
   ```
3. Set up your Binance API keys in the script:
   ```python
   API_KEY = "your_api_key"
   API_SECRET = "your_api_secret"
   ```

## Configuration
The `TRADING_CONFIG` dictionary allows you to configure trade parameters for different pairs:
```python
TRADING_CONFIG = {
    "DOGEUSDT": {"trade_size": 50, "sl_adjust": 0.002, "tp_adjust": 0.005},
    "ETHUSDT": {"trade_size": 0.01, "sl_adjust": 5, "tp_adjust": 15},
    "BTCUSDT": {"trade_size": 0.001, "sl_adjust": 100, "tp_adjust": 300}
}
```
Modify `TRADING_PAIR` to change the active trading pair:
```python
TRADING_PAIR = "DOGEUSDT"  # Change to ETHUSDT or BTCUSDT as needed
```

## How It Works
1. **Market Analysis**
   - Fetches historical candlestick (kline) data.
   - Computes MA3, MA9, and MA21.
   - Identifies support and resistance levels.
   - Determines trade signals based on price action.

2. **Trade Execution**
   - Buys if price bounces above MA3 and holds above MA9.
   - Sets stop-loss (SL) below support.
   - Sets take-profit (TP) above resistance.

3. **Monitoring & Exit**
   - Continuously checks price movements.
   - Sells when TP or SL is triggered.

## Usage
Run the script with:
```bash
python scalping_bot.py
```
The bot will analyze the market and execute trades automatically.

## Disclaimer
This script is for educational purposes only. Trading cryptocurrencies carries risks, and you should use this bot at your own discretion. Always test in a simulated environment before deploying real funds.

## License
MIT License
