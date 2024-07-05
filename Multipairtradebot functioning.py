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
desired_profit_margin = 0.0025  # Desired profit margin of 0.25%
buy_prices = {}  # Dictionary to store buy prices and amounts for each pair
model = None
scaler = None

# Initialize trade log DataFrame
trade_log_df = pd.DataFrame(columns=[
    'Trade ID', 'Timestamp', 'Market', 'Type', 'Quantity', 'Price', 'Total', 'Fee', 'Associated Trade IDs', 'Profit/Loss'
])
trade_id = 1

# Function to get the minimum notional value for a trading pair
def get_min_notional_value(pair):
    try:
        markets = binance.load_markets()
        min_notional = markets[pair]['limits']['cost']['min']
        return min_notional
    except Exception as e:
        logging.error(f"Error fetching minimum notional value for {pair}: {e}")
        return None

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

@retry_with_time_sync()
async def get_last_order(pair, side):
    try:
        orders = binance.fetch_my_trades(pair)
        if not orders:
            return None
        
        # Filter the orders by side and get the last one
        filtered_orders = [order for order in orders if order['side'] == side]
        if not filtered_orders:
            return None
        
        last_order = max(filtered_orders, key=lambda x: x['timestamp'])
        return last_order['price'], last_order['amount']
    except Exception as e:
        logging.error(f"Error fetching last {side} order for {pair}: {e}")
        return None, None

# Function to check if a market pair is valid
def is_valid_pair(pair):
    try:
        markets = binance.load_markets()
        if pair in markets:
            return True
        else:
            logging.warning(f"Market pair {pair} is not valid.")
            return False
    except Exception as e:
        logging.error(f"Error verifying market pair {pair}: {e}")
        return False

# Function to place orders
@retry_with_time_sync()
async def place_order(pair, side, amount, price):
    try:
        synchronize_time()  # Synchronize time before making the API call
        if not is_valid_pair(pair):
            return None
        min_notional = get_min_notional_value(pair)
        if min_notional is None or amount * price < min_notional:
            logging.warning(f"Order value {amount * price} for {pair} is below the minimum notional value {min_notional}")
            return None
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

# Function to fetch real-time prices for all assets in USDC
async def get_realtime_prices():
    prices = {}
    for asset in ['USDC', 'ETH', 'BTC', 'SOL']:
        if asset == 'USDC':
            prices[asset] = 1.0
        else:
            ticker = binance.fetch_ticker(f'{asset}/USDC')
            prices[asset] = ticker['last']
    return prices

# Function to calculate the net value of holdings in USDC
async def calculate_net_value():
    prices = await get_realtime_prices()
    net_value = sum(balances[asset] * prices[asset] for asset in balances)
    return net_value

# Function to log trades and calculate profit/loss
def log_trade(trade_id, timestamp, pair, side, amount, price, total, fee, associated_trade_ids='', profit_loss=None):
    global trade_log_df
    new_trade = pd.DataFrame([{
        'Trade ID': trade_id,
        'Timestamp': timestamp,
        'Market': pair,
        'Type': side.upper(),
        'Quantity': amount,
        'Price': price,
        'Total': total,
        'Fee': fee,
        'Associated Trade IDs': associated_trade_ids,
        'Profit/Loss': profit_loss
        }])

    trade_log_df = pd.concat([trade_log_df, new_trade], ignore_index=True)

# Function to save trade log to CSV
def save_trade_log():
    trade_log_df.to_csv('Multipairtrade_log.csv', index=False)

# Function to execute trade based on signal
async def execute_trade(pair, signal, amount, price, atr, indicators):
    global balances, trade_log, buy_prices, trade_id
    try:
        # Fetch the updated balance before placing orders
        await update_balance()

        initial_net_value = await calculate_net_value()

        if signal == 'buy':
            available_balance = balances[pair.split('/')[1]]
            trade_amount = available_balance / (price * (1 + maker_fee))
            if trade_amount < minimum_amount:
                logging.warning(f"Insufficient {pair.split('/')[1]} balance to buy minimum amount of {pair.split('/')[0]}. Required: {minimum_amount} {pair.split('/')[0]}")
                return
            total_cost = trade_amount * price
            fee = total_cost * maker_fee
            log_trade(trade_id, datetime.now(), pair, 'buy', trade_amount, price, total_cost, fee)
            trade_id += 1
            if pair not in buy_prices:
                buy_prices[pair] = []
            buy_prices[pair].append((price, trade_amount))
        else:
            available_balance = balances[pair.split('/')[0]]
            trade_amount = available_balance
            if trade_amount < minimum_amount:
                logging.warning(f"Trade amount {trade_amount} {pair.split('/')[0]} does not meet the minimum requirements")
                return

            # Fetch the last buy price
            last_buy_price, _ = await get_last_order(pair, 'buy')
            if not last_buy_price:
                logging.warning(f"No buy prices recorded for {pair}. Cannot execute sell.")
                return

            # Calculate the necessary sell price to ensure profit
            min_sell_price = last_buy_price * (1 + maker_fee) * (1 + taker_fee) * (1 + desired_profit_margin)
            if price < min_sell_price:
                logging.warning(f"Sell price {price} is not sufficient to cover fees and desired profit margin. Required: {min_sell_price}")
                return

            effective_sold_amount = trade_amount * (1 - taker_fee)
            sell_total = trade_amount * price
            fee = sell_total * taker_fee
            buy_total = sum([amount * buy_price for buy_price, amount in buy_prices.get(pair, [])])
            profit_loss = sell_total - fee - buy_total

            # Remove the matched buy prices
            remaining_amount = trade_amount
            for buy_price, buy_amount in buy_prices[pair][:]:
                if remaining_amount <= 0:
                    break
                if buy_amount <= remaining_amount:
                    remaining_amount -= buy_amount
                    buy_prices[pair].remove((buy_price, buy_amount))
                else:
                    new_amount = buy_amount - remaining_amount
                    buy_prices[pair].remove((buy_price, buy_amount))
                    buy_prices[pair].append((buy_price, new_amount))
                    remaining_amount = 0

        order = await place_order(pair, signal, trade_amount, price)
        if order:
            logging.info(f"Order placed for {pair}: {order}")
            await asyncio.sleep(2)
            order_status = binance.fetch_order(order['id'], pair)
            market_price = binance.fetch_ticker(pair)['last']  # Fetch the latest market price
            if order_status['status'] == 'open':
                # Order replacement logic
                new_prices = await get_historical_prices(pair)
                new_indicators = calculate_indicators(new_prices)
                new_order_flow_imbalance = await calculate_order_flow_imbalance(pair)
                new_indicators['order_flow_imbalance'] = new_order_flow_imbalance

                new_rsi = new_indicators['rsi']
                new_macd_line = new_indicators['macd_line']
                new_macd_signal = new_indicators['macd_signal']
                new_bb_upper = new_indicators['bb_upper']
                new_bb_lower = new_indicators['bb_lower']
                new_atr = new_indicators['atr']

                new_market_price = new_prices['close'][-1]

                if signal == 'buy':
                    if new_market_price < order['price'] * 0.98:
                        logging.info(f"Replacing buy order for {pair} due to price drop: Order Price: {order['price']}, New Market Price: {new_market_price}")
                        await replace_order(pair, order['id'], signal, trade_amount, new_market_price)
                    elif new_rsi < 25 or new_macd_line < new_macd_signal:
                        logging.info(f"Canceling buy order for {pair} due to unfavorable indicators: RSI: {new_rsi}, MACD Line: {new_macd_line}, MACD Signal: {new_macd_signal}")
                        await binance.cancel_order(order['id'], pair)
                elif signal == 'sell':
                    if new_market_price > order['price'] * 1.02:
                        logging.info(f"Replacing sell order for {pair} due to price rise: Order Price: {order['price']}, New Market Price: {new_market_price}")
                        await replace_order(pair, order['id'], signal, trade_amount, new_market_price)
                    elif new_rsi > 75 or new_macd_line > new_macd_signal:
                        logging.info(f"Canceling sell order for {pair} due to unfavorable indicators: RSI: {new_rsi}, MACD Line: {new_macd_line}, MACD Signal: {new_macd_signal}")
                        await binance.cancel_order(order['id'], pair)
            elif order_status['status'] == 'closed':
                filled_amount = order_status['filled']
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

        final_net_value = await calculate_net_value()
        profit_loss = final_net_value - initial_net_value
        logging.info(f"Trade ID: {trade_id}, Pair: {pair}, Signal: {signal}, Initial Net Value: {initial_net_value}, Final Net Value: {final_net_value}, Profit/Loss: {profit_loss}")

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
                await execute_trade(pair, 'buy', 0, market_price, atr, indicators)  # Set amount to 0 for initial call

            if sell_signal or ml_sell_signal:
                logging.info(f"Sell signal generated for {pair} at price: {market_price}")
                await execute_trade(pair, 'sell', 0, market_price, atr, indicators)  # Set amount to 0 for initial call

        save_trade_log()  # Save the trade log periodically
        await asyncio.sleep(60)

if __name__ == '__main__':
    asyncio.run(main())

# Graceful shutdown handling
def handle_exit(signum, frame):
    logging.info("Shutting down the bot...")
    save_trade_log()  # Save the trade log before shutting down
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
