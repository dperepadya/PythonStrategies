# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from binance.client import Client
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, cross, resample_apply, plot_heatmaps

enable_echo = True

def load_data(symbol, interval, start_date, end_date):
    """
    Load OHLCV MD for a symbol
    """
    klines = client.get_historical_klines(symbol, interval, start_date, end_date)
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades',
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    df = pd.DataFrame(klines, columns=columns)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df.set_index('timestamp', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    return df


def load_data_for_symbols(symbols, interval, start_date, end_date):
    """
    Load OHLCV MD for a symbols batch
    """
    data = {}
    for symbol in symbols:
        df = load_data(symbol, interval, start_date, end_date)
        data[symbol] = df
    return data

def log(*args, **kwargs):
    if enable_echo:
        print(*args, **kwargs)

def keltner_channel(data, period=12, atr_multiplier=2):
    """
    Keltner Channel Indicator (ATR volatility range)
    """
    true_range = data['High'] - data['Low']
    # print(len(true_range))
    previous_close = pd.Series(data['Close']).shift(1)

    high_minus_previous_close = abs(data['High'] - previous_close)
    low_minus_previous_close = abs(data['Low'] - previous_close)

    true_range = np.maximum.reduce([true_range, high_minus_previous_close, low_minus_previous_close])

    true_range_series = pd.Series(true_range, index=data.index)
    # print(len(true_range_series))

    true_range_series = pd.Series(data['High']) - pd.Series(data['Low'])
    # print(len(true_range_series))
    # Compute ATR
    atr = true_range_series.rolling(period, min_periods=1).mean()
    # print('atr', len(atr))
    # Compute Keltner Channel range
    mean = pd.Series(data['Close']).rolling(period).mean()
    upper_channel = mean + atr * atr_multiplier
    lower_channel = mean - atr * atr_multiplier
    # print(len(upper_channel))
    return pd.DataFrame({'upper': upper_channel, 'lower': lower_channel, 'mean': mean})


class KelnerChannelRVStrategy(Strategy):
    """
    Simple RV trading system based on Keltner channel.
    Entry Logic: Price is outside a channel's border
    Exit Logic: Price returned to the channel's mean
    Position Management: Stop Loss, Take Profit, Breakeven
    """
    # Logic

    # Input parameters for Keltner Channels Spread
    kch_period = 6
    kch_mult = 3.5

    # Exit trigger
    exit_trig_is_on = 0
    # stop_loss
    stop_loss_is_on = 1
    # Extremum search lookback
    stop_loss_lookback = 4
    stop_loss_pct = 0.0
    # Break even from Entry price
    break_even_is_on = 1
    break_even_pct = 0.001
    # Dynamic TP
    profit_target_is_on = 0
    profit_target_dyn_ratio = 1

    def init(self):
        super().init()
        # Compute fast and slow Kelner Channels
        log(type(self.data.Close))
        self.kch_channel = self.I(keltner_channel, self.data, self.kch_period, self.kch_mult)

        self.kch_upborder = self.kch_channel[0]
        self.kch_downborder = self.kch_channel[1]
        self.kch_mean = self.kch_channel[2]

        log('up fast', self.kch_upborder)
        log('dn fast', self.kch_downborder)

        # Position varaibles
        self.stop_loss = 0
        self.profit_target = 0
        self.be_level = 0
        self.entry_price = 0

    def next(self):
        # A Position is opened
        if self.position:
            log(self.position.size)
            # Check Exit triggers
            exit_trigger = False
            sign = 1 if self.position.is_long else -1
            side = 'long' if self.position.is_long else 'short'
            # Check Exit trigger
            exit_trigger = (self.data.Close[-1] - self.kch_mean[-1]) * sign > 0 or \
                           self.stop_loss > 0 and (self.data.Close[-1] - self.stop_loss) * sign < 0 or \
                           self.profit_target > 0 and (self.data.Close[-1] - self.profit_target) * sign > 0
            if exit_trigger:
                log(side, 'close trigger:', 'mean', self.kch_mean[-1], self.data.Close[-1])
                # Exit action
                # log('close the position:', self.position.size, self.data.index[-1], self.data.Close[-1])
                # log('-' * 90)
                self.position.close()
            # Check Breakeven trigger
            elif self.break_even_is_on == 1 and self.be_level == 0:
                thresh = self.entry_price * (1 + self.break_even_pct * sign)
                if (self.data.Close[-1] - thresh) * sign > 0:
                    self.be_level = thresh
                    self.stop_loss = thresh
                    # log('be level:', self.data.index[-1], self.entry_price, self.break_even_pct, '->', thresh, self.data.Close[-1])

        # Is Flat
        # log(self.position, type(self.position))
        if self.position.size == 0:
            # log('position is flat')
            # Long trigger
            if self.data.Close[-1] < self.kch_downborder[-1]:
                log('long trigger', self.data.index[-1], 'c', self.data.Close[-1], 'o', self.data.Open[-1], 'dn',
                    self.kch_downborder[-1])
                log('open long', self.data.Close[-1])
                self.buy()
                self.entry_price = self.data.Close[-1]
                self.be_level = 0
                # Set initial Stop Loss and Profit Target
                if self.stop_loss_is_on == 1:
                    min_level = pd.Series(self.data.Low).rolling(window=self.stop_loss_lookback).min().iloc[-1]
                    # log(min_level)
                    self.stop_loss = min_level * (1 - self.stop_loss_pct)
                # log('sl', self.stop_loss)
                if self.profit_target_is_on == 1:
                    self.profit_target = self.entry_price + abs(
                        self.entry_price - self.stop_loss) * self.profit_target_dyn_ratio
                log(f'intial sl {self.stop_loss} and tp {self.profit_target}')
            # Short trigger
            elif self.data.Close[-1] > self.kch_upborder[-1]:
                log('short trigger', self.data.index[-1], 'c', self.data.Close[-1], 'o', self.data.Open[-1], 'up',
                    self.kch_upborder[-1])
                log('open short', self.data.Close[-1])
                self.sell()
                self.entry_price = self.data.Close[-1]
                self.be_level = 0
                # Set initial Stop Loss and Profit Target
                if self.stop_loss_is_on == 1:
                    max_level = pd.Series(self.data.High).rolling(window=self.stop_loss_lookback).max().iloc[-1]
                    # log(max_level)
                    self.stop_loss = max_level * (1 + self.stop_loss_pct)
                    # log('sl', self.stop_loss)
                if self.profit_target_is_on == 1:
                    self.profit_target = self.entry_price - abs(
                        self.entry_price - self.stop_loss) * self.profit_target_dyn_ratio
                log(f'intial sl {self.stop_loss} and tp {self.profit_target}')


if __name__ == '__main__':
    api_key = '123'
    api_secret = '123'
    client = Client(api_key, api_secret)
    symbols = ['BTCUSDT']
    interval = Client.KLINE_INTERVAL_15MINUTE
    start_date = '2023-01-01'
    end_date = '2023-06-30'
    symbols_df = load_data_for_symbols(symbols, interval, start_date, end_date)
    symbol = 'BTCUSDT'
    df = symbols_df[symbol]
    print(df)
    fee_per_turn = 0.001
    deposit = 100000
    bt = Backtest(df, KelnerChannelRVStrategy, cash=deposit, commission=fee_per_turn, trade_on_close=True,
                  hedging=False, exclusive_orders=True)
    stats = bt.run()
    print(stats)