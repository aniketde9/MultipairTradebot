import ccxt
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import ta
import pygame
import time
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load configuration from external file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Configure logging
logging.basicConfig(filename=config['logging']['file'], level=getattr(logging, config['logging']['level']), format=config['logging']['format'])

# Initialize Binance API
binance = ccxt.binance({
    'apiKey': config['binance']['apiKey'],
    'secret': config['binance']['secret'],
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot',
        'adjustForTimeDifference': True,
        'recvWindow': 60000,
    }
})

# Pairs to trade
pairs = config['trading']['pairs']

# Global Variables
balances = {}
minimum_amount = 0.001
trade_log = []
maker_fee = 0.001  # Maker fee
taker_fee = 0.001  # Taker fee
buy_prices = {}  # Dictionary to store buy prices and amounts for each pair
model = None
scaler = None

# Synchronize with Binance server time
def synchronize_time():
    try:
        server_time = binance.fetch_time()
        local_time = int(datetime.now().timestamp() * 1000)
        binance.options['adjustForTimeDifference'] = True
        binance.options['timeDifference'] = server_time - local_time
        logging.info(f"Synchronized server time. Server time: {server_time}, Local time: {local_time}, Time difference: {binance.options['timeDifference']}")
    except Exception as e:
        logging.error(f"Error synchronizing time: {e}")

# Decorator for retrying API calls indefinitely with a fixed delay
def retry_with_time_sync(delay=5):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            while True:
                try:
                    return await func(*args, **kwargs)
                except ccxt.BaseError as e:
                    if 'Timestamp for this request was 1000ms ahead of the server\'s time' in str(e):
                        logging.warning(f"Timestamp error: {e}. Synchronizing time and retrying...")
                        synchronize_time()
                    else:
                        logging.error(f"API Error: {e}")
                await asyncio.sleep(delay)
        return wrapper
    return decorator

# Function to get the initial balance for specific assets
@retry_with_time_sync()
async def get_initial_balance():
    global balances
    balance = binance.fetch_balance()
    for asset in ['USDC', 'ETH', 'BTC', 'SOL']:
        if asset in balance['total']:
            balances[asset] = balance['total'][asset]
        else:
            balances[asset] = 0.0
    logging.info(f"Initial balances: {balances}")
    print(f"Initial balances: {balances}")

# Fetch historical prices for indicator calculation
@retry_with_time_sync()
async def get_historical_prices(pair):
    ohlcv = binance.fetch_ohlcv(pair, timeframe='1m', limit=100)
    prices = {
        'timestamp': [x[0] for x in ohlcv],
        'open': [x[1] for x in ohlcv],
        'high': [x[2] for x in ohlcv],
        'low': [x[3] for x in ohlcv],
        'close': [x[4] for x in ohlcv],
        'volume': [x[5] for x in ohlcv]
    }
    return prices

# Function to calculate technical indicators using ta
def calculate_indicators(prices):
    close_prices = pd.Series(prices['close'])
    high_prices = pd.Series(prices['high'])
    low_prices = pd.Series(prices['low'])

    rsi = ta.momentum.RSIIndicator(close=close_prices, window=14).rsi()
    macd = ta.trend.MACD(close=close_prices)
    macd_line = macd.macd()
    macd_signal = macd.macd_signal()
    bb = ta.volatility.BollingerBands(close=close_prices, window=20, window_dev=2)
    atr = ta.volatility.AverageTrueRange(high=high_prices, low=low_prices, close=close_prices, window=14).average_true_range()

    indicators = {
        'timestamp': datetime.now(),
        'price': close_prices.iloc[-1],
        'rsi': rsi.iloc[-1],
        'macd_line': macd_line.iloc[-1],
        'macd_signal': macd_signal.iloc[-1],
        'bb_upper': bb.bollinger_hband().iloc[-1],
        'bb_lower': bb.bollinger_lband().iloc[-1],
        'atr': atr.iloc[-1]
    }
    return indicators

# Function to calculate order flow imbalance
@retry_with_time_sync()
async def calculate_order_flow_imbalance(pair):
    order_book = binance.fetch_order_book(pair)
    total_bids = sum([bid[1] for bid in order_book['bids']])
    total_asks = sum([ask[1] for ask in order_book['asks']])
    order_flow_imbalance = (total_bids - total_asks) / (total_bids + total_asks)
    return order_flow_imbalance

# Function to place orders
@retry_with_time_sync()
async def place_order(pair, side, amount, price):
    try:
        synchronize_time()  # Synchronize time before making the API call
        if side == 'buy':
            order = binance.create_limit_buy_order(pair, amount, price)
        else:
            order = binance.create_limit_sell_order(pair, amount, price)
        logging.info(f"Placed {side} order for {pair}: {order}")
        return order
    except ccxt.BaseError as e:
        logging.error(f"API Error while placing order for {pair}: {e}")
    return None

# Function to replace open orders if certain conditions are met
@retry_with_time_sync()
async def replace_order(pair, order_id, side, amount, new_price):
    try:
        synchronize_time()  # Synchronize time before making the API call
        binance.cancel_order(order_id, pair)
        logging.info(f"Canceled open order for {pair}: {order_id}")
        return await place_order(pair, side, amount, new_price)
    except ccxt.BaseError as e:
        logging.error(f"API Error while replacing order for {pair}: {e}")
    return None

# Dynamic signal generation logic
async def dynamic_signal_generation(pair):
    prices = await get_historical_prices(pair)
    indicators = calculate_indicators(prices)
    market_price = prices['close'][-1]
    order_flow_imbalance = await calculate_order_flow_imbalance(pair)
    indicators['order_flow_imbalance'] = order_flow_imbalance

    rsi = indicators['rsi']
    macd_line = indicators['macd_line']
    macd_signal = indicators['macd_signal']
    bb_upper = indicators['bb_upper']
    bb_lower = indicators['bb_lower']
    atr = indicators['atr']

    logging.info(f"Pair: {pair}, RSI: {rsi}, MACD Line: {macd_line}, MACD Signal: {macd_signal}, BB Upper: {bb_upper}, BB Lower: {bb_lower}, ATR: {atr}, Order Flow Imbalance: {order_flow_imbalance}")

    buy_signal = (macd_line > macd_signal and 25 < rsi < 45 and market_price <= bb_lower * 1.25 and order_flow_imbalance > 0)
    sell_signal = (macd_line < macd_signal - 0.2 and 75 > rsi > 55 and market_price >= bb_upper * 0.85 and order_flow_imbalance < 0)

    return buy_signal, sell_signal, market_price, atr, indicators

# Function to execute trade based on signal
async def execute_trade(pair, signal, amount, price, atr, indicators):
    global balances, trade_log, buy_prices
    try:
        # Fetch the updated balance before placing orders
        await update_balance()

        if signal == 'buy':
            available_balance = balances[pair.split('/')[1]]
            trade_amount = available_balance / (price * (1 + maker_fee))
            if trade_amount < minimum_amount:
                logging.warning(f"Insufficient {pair.split('/')[1]} balance to buy minimum amount of {pair.split('/')[0]}. Required: {minimum_amount} {pair.split('/')[0]}")
                return
        else:
            available_balance = balances[pair.split('/')[0]]
            trade_amount = available_balance
            if trade_amount < minimum_amount:
                logging.warning(f"Trade amount {trade_amount} {pair.split('/')[0]} does not meet the minimum requirements")
                return

            effective_sold_amount = trade_amount * (1 - taker_fee)
            profit = effective_sold_amount * price - available_balance
            if profit <= 0:
                logging.warning(f"No profit would be made after fees. Trade amount: {trade_amount}, Effective sold amount: {effective_sold_amount}, Profit: {profit}")
                return

        order = await place_order(pair, signal, trade_amount, price)
        if order:
            logging.info(f"Order placed for {pair}: {order}")
            await asyncio.sleep(2)
            order_status = binance.fetch_order(order['id'], pair)
            market_price = binance.fetch_ticker(pair)['last']  # Fetch the latest market price
            if order_status['status'] == 'open':
                if signal == 'buy' and market_price < order['price'] * 0.98:
                    logging.info(f"Replacing buy order for {pair} due to price drop: Order Price: {order['price']}, Market Price: {market_price}")
                    await replace_order(pair, order['id'], signal, trade_amount, market_price)
                elif signal == 'sell' and market_price > order['price'] * 1.02:
                    logging.info(f"Replacing sell order for {pair} due to price rise: Order Price: {order['price']}, Market Price: {market_price}")
                    await replace_order(pair, order['id'], signal, trade_amount, market_price)
            elif order_status['status'] == 'closed':
                filled_amount = order_status['filled']
                if signal == 'buy':
                    if pair not in buy_prices:
                        buy_prices[pair] = []
                    buy_prices[pair].append((price, filled_amount))  # Store buy price and amount
                elif signal == 'sell':
                    valid_sell = False
                    if pair in buy_prices:
                        for buy_price, buy_amount in buy_prices[pair]:
                            if price > buy_price * (1 + maker_fee):  # Ensure selling price is higher than the buy price accounting for fees
                                valid_sell = True
                                buy_prices[pair].remove((buy_price, buy_amount))  # Remove the matched buy price after sell
                                break
                    if not valid_sell:
                        logging.info(f"Skipping sell for {pair} due to no profitable buy price found. Market Price: {market_price}")
                        return
                trade_log.append({
                    'pair': pair,
                    'timestamp': datetime.now(),
                    'side': signal,
                    'amount': filled_amount,
                    'price': price,
                    'status': 'filled',
                    'rsi': indicators['rsi'],
                    'macd_line': indicators['macd_line'],
                    'macd_signal': indicators['macd_signal'],
                    'bb_upper': indicators['bb_upper'],
                    'bb_lower': indicators['bb_lower'],
                    'atr': indicators['atr'],
                    'usdc_balance': balances['USDC'],
                    'sol_balance': balances['SOL'],
                })
                logging.info(f"Updated balances for {pair} - USDC: {balances['USDC']}, SOL: {balances['SOL']}")
            else:
                logging.info(f"Order not filled for {pair}: {order_status}")
    except Exception as e:
        logging.error(f"Error executing trade for {pair}: {e}")

# Function to fetch updated balance and ensure only initial balance is used for trading
@retry_with_time_sync()
async def update_balance():
    global balances
    try:
        synchronize_time()  # Synchronize time before making the API call
        balance = binance.fetch_balance()
        for asset in ['USDC', 'ETH', 'BTC', 'SOL']:
            if asset in balance['total']:
                balances[asset] = balance['total'][asset]
            else:
                balances[asset] = 0.0
        logging.info(f"Updated balances: {balances}")
    except Exception as e:
        logging.error(f"Error fetching balance: {e}")

# Function to log indicators and market data to a file
def log_data_to_file(pair, indicators, market_price):
    logging.info(f"Pair: {pair}, Market Price: {market_price}")
    logging.info(f"Indicators: {indicators}")

# Machine learning model for signal generation
def train_ml_model():
    global model, scaler
    # Load historical data and calculate indicators
    # For simplicity, we're generating random data here
    data = pd.DataFrame({
        'price': np.random.randn(1000),
        'rsi': np.random.randn(1000),
        'macd_line': np.random.randn(1000),
        'macd_signal': np.random.randn(1000),
        'bb_upper': np.random.randn(1000),
        'bb_lower': np.random.randn(1000),
        'atr': np.random.randn(1000),
        'order_flow_imbalance': np.random.randn(1000),
        'signal': np.random.randint(2, size=1000)  # Random buy/sell signals
    })
    
    # Feature scaling
    scaler = StandardScaler()
    features = data[['price', 'rsi', 'macd_line', 'macd_signal', 'bb_upper', 'bb_lower', 'atr', 'order_flow_imbalance']]
    features_scaled = scaler.fit_transform(features)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, data['signal'], test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    logging.info(f"Model training complete. Accuracy: {accuracy}")

# Function to generate buy/sell signals using the machine learning model
def generate_ml_signal(indicators):
    global model, scaler
    if model is None:
        logging.warning("ML model is not trained. Skipping ML signal generation.")
        return None, None

    # Extract features and ensure they have the same structure as during training
    features = pd.DataFrame([[
        indicators['price'],
        indicators['rsi'],
        indicators['macd_line'],
        indicators['macd_signal'],
        indicators['bb_upper'],
        indicators['bb_lower'],
        indicators['atr'],
        indicators['order_flow_imbalance']
    ]], columns=['price', 'rsi', 'macd_line', 'macd_signal', 'bb_upper', 'bb_lower', 'atr', 'order_flow_imbalance'])

    features_scaled = scaler.transform(features)
    
    # Predict buy/sell signal
    signal = model.predict(features_scaled)
    buy_signal = signal == 1
    sell_signal = signal == 0
    return buy_signal, sell_signal

# Main function
async def main():
    synchronize_time()  # Ensure time is synchronized before starting
    await get_initial_balance()
    train_ml_model()  # Train the machine learning model
    while True:
        await update_balance()
        tasks = [dynamic_signal_generation(pair) for pair in pairs]
        results = await asyncio.gather(*tasks)

        for i, pair in enumerate(pairs):
            buy_signal, sell_signal, market_price, atr, indicators = results[i]
            ml_buy_signal, ml_sell_signal = generate_ml_signal(indicators)
            
            log_data_to_file(pair, indicators, market_price)

            if buy_signal or ml_buy_signal:
                logging.info(f"Buy signal generated for {pair} at price: {market_price}")
                asyncio.create_task(execute_trade(pair, 'buy', 0, market_price, atr, indicators))  # Set amount to 0 for initial call

            if sell_signal or ml_sell_signal:
                logging.info(f"Sell signal generated for {pair} at price: {market_price}")
                asyncio.create_task(execute_trade(pair, 'sell', 0, market_price, atr, indicators))  # Set amount to 0 for initial call

        await asyncio.sleep(60)

if __name__ == '__main__':
    asyncio.run(main())

# Graceful shutdown handling
def handle_exit(signum, frame):
    logging.info("Shutting down the bot...")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.sleep(1))
    loop.close()
    exit(0)

# Signal handling for graceful shutdown
import signal
signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

# Path to your alert sound file (ensure the file exists at this path)
alert_sound_path = 'D:\\Downloads\\Loud-Music.mp3'

# Initialize pygame mixer
pygame.mixer.init()

def play_alert_sound():
    pygame.mixer.music.load(alert_sound_path)
    pygame.mixer.music.play()

try:
    # Your main code goes here
    # For demonstration, we'll use a simple example
    while True:
        print("Running...")
        time.sleep(2)  # Simulating some ongoing process
        # Simulate an error after some time
        if time.time() % 10 < 0.1:
            raise Exception("Simulated error")

except Exception as e:
    print(f"An error occurred: {e}")
    play_alert_sound()
    # Keep the program running long enough to hear the sound
    time.sleep(0)  # Adjust the time as needed
