
from flask import Flask, render_template, request
import requests
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
from scipy.signal import find_peaks
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import mplfinance as mpf
from rolling_window import rw_top, rw_bottom
from typing import List
from collections import deque
from dataclasses import dataclass
import uuid
app = Flask(__name__)
@dataclass
class HSPattern:
    inverted: bool
    l_shoulder: int = -1
    r_shoulder: int = -1
    l_armpit: int = -1
    r_armpit: int = -1
    head: int = -1
    l_shoulder_p: float = -1
    r_shoulder_p: float = -1
    l_armpit_p: float = -1
    r_armpit_p: float = -1
    head_p: float = -1
    start_i: int = -1
    break_i: int = -1
    break_p: float = -1
    neck_start: float = -1
    neck_end: float = -1
    neck_slope: float = -1
    head_width: float = -1
    head_height: float = -1
    pattern_r2: float = -1

def compute_pattern_r2(data: np.array, pat: HSPattern):
    # ... (unchanged)
    line0_slope = (pat.l_shoulder_p - pat.neck_start) / (pat.l_shoulder - pat.start_i)
    line0 = pat.neck_start + np.arange(pat.l_shoulder - pat.start_i) * line0_slope
    
    line1_slope = (pat.l_armpit_p - pat.l_shoulder_p) / (pat.l_armpit - pat.l_shoulder)
    line1 = pat.l_shoulder_p + np.arange(pat.l_armpit - pat.l_shoulder) * line1_slope
    
    line2_slope = (pat.head_p - pat.l_armpit_p) / (pat.head - pat.l_armpit)
    line2 = pat.l_armpit_p + np.arange(pat.head - pat.l_armpit) * line2_slope
    
    line3_slope = (pat.r_armpit_p - pat.head_p) / (pat.r_armpit - pat.head)
    line3 = pat.head_p + np.arange(pat.r_armpit - pat.head) * line3_slope
    
    line4_slope = (pat.r_shoulder_p - pat.r_armpit_p) / (pat.r_shoulder - pat.r_armpit)
    line4 = pat.r_armpit_p + np.arange(pat.r_shoulder - pat.r_armpit) * line4_slope
    
    line5_slope = (pat.break_p - pat.r_shoulder_p) / (pat.break_i - pat.r_shoulder)
    line5 = pat.r_shoulder_p + np.arange(pat.break_i - pat.r_shoulder) * line5_slope
    
    raw_data = data[pat.start_i:pat.break_i]
    hs_model = np.concatenate([line0, line1, line2, line3, line4, line5])
    mean = np.mean(raw_data)

    ss_res = np.sum( (raw_data - hs_model) ** 2.0 )
    ss_tot = np.sum( (raw_data - mean) ** 2.0 )

    r2 = 1.0 - ss_res / ss_tot
    return r2

def check_hs_pattern(extrema_indices: List[int], data: np.array, i: int, early_find: bool = False) -> HSPattern:
    # ... (unchanged)
    l_shoulder = extrema_indices[0]
    l_armpit = extrema_indices[1]
    head = extrema_indices[2]
    r_armpit = extrema_indices[3]

    if i - r_armpit < 2:
        return None

    # Find right shoulder as max price since r_armpit
    r_shoulder = r_armpit + data[r_armpit + 1: i].argmax() + 1

    # Head must be higher than shoulders
    if data[head] <= max(data[l_shoulder], data[r_shoulder]):
        return None

    # Balance rule. Shoulders are above the others midpoint.
    # A shoulder's midpoint is the midpoint between the shoulder and armpit
    r_midpoint = 0.5 * (data[r_shoulder] + data[r_armpit])
    l_midpoint = 0.5 * (data[l_shoulder] + data[l_armpit])
    if data[l_shoulder] < r_midpoint  or data[r_shoulder] < l_midpoint:
        return None

    # Symmetry rule. time from shoulder to head are comparable
    r_to_h_time = r_shoulder - head
    l_to_h_time = head - l_shoulder
    if r_to_h_time > 2.5 * l_to_h_time or l_to_h_time > 2.5 * r_to_h_time:
        return None
        
    # Compute neckline
    neck_run = r_armpit - l_armpit
    neck_rise = data[r_armpit] - data[l_armpit]
    neck_slope = neck_rise / neck_run
    
    # neckline value at current index
    neck_val = data[l_armpit] + (i - l_armpit) * neck_slope

    # Confirm pattern when price is halfway from right shoulder
    if early_find: 
        if data[i] > r_midpoint:
            return None
    else:
       
        # Price has yet to break neckline, unconfirmed
        if data[i] > neck_val:
            return None

    # Find beginning of pattern. Neck to left shoulder
    head_width = r_armpit - l_armpit
    pat_start = -1
    neck_start = -1
    for j in range(1, head_width):
        neck = data[l_armpit] + (l_shoulder - l_armpit - j) * neck_slope
        
        if l_shoulder - j < 0:
            return None
        
        if data[l_shoulder - j] < neck:
            pat_start = l_shoulder - j
            neck_start = neck
            break

    if pat_start == -1:
        return None

    # Pattern confirmed if here :)
    pat = HSPattern(inverted=False)  
    
    pat.l_shoulder = l_shoulder
    pat.r_shoulder = r_shoulder
    pat.l_armpit = l_armpit
    pat.r_armpit = r_armpit
    pat.head = head
    
    pat.l_shoulder_p = data[l_shoulder]
    pat.r_shoulder_p = data[r_shoulder]
    pat.l_armpit_p = data[l_armpit]
    pat.r_armpit_p = data[r_armpit]
    pat.head_p = data[head]

    pat.start_i = pat_start
    pat.break_i = i
    pat.break_p = data[i]

    pat.neck_start = neck_start
    pat.neck_end = neck_val

    pat.neck_slope = neck_slope
    pat.head_width = head_width
    pat.head_height = data[head] - (data[l_armpit] + (head - l_armpit) * neck_slope)
    pat.pattern_r2 = compute_pattern_r2(data, pat)

    # I experiemented with r-squared as a filter for H&S, but this can delay recognition.
    # It didn't seem terribly potent, may be useful as a filter in conjunction with other attributes
    # if one wanted to add a machine learning layer before trading these patterns. 

    #if pat.pattern_r2 < 0.0:
    #    return None

    return pat

def check_ihs_pattern(extrema_indices: List[int], data: np.array, i: int, early_find: bool = False) -> HSPattern:
    # ... (unchanged)
    l_shoulder = extrema_indices[0]
    l_armpit = extrema_indices[1]
    head = extrema_indices[2]
    r_armpit = extrema_indices[3]
    
    if i - r_armpit < 2:
        return None

    # Find right shoulder as max price since r_armpit
    r_shoulder = r_armpit + data[r_armpit+1: i].argmin() + 1

    # Head must be lower than shoulders
    if data[head] >= min(data[l_shoulder], data[r_shoulder]):
        return None

    # Balance rule. Shoulders are below the others midpoint.
    # A shoulder's midpoint is the midpoint between the shoulder and armpit
    r_midpoint = 0.5 * (data[r_shoulder] + data[r_armpit])
    l_midpoint = 0.5 * (data[l_shoulder] + data[l_armpit])
    if data[l_shoulder] > r_midpoint  or data[r_shoulder] > l_midpoint:
        return None

    # Symmetry rule. time from shoulder to head are comparable
    r_to_h_time = r_shoulder - head
    l_to_h_time = head - l_shoulder
    if r_to_h_time > 2.5 * l_to_h_time or l_to_h_time > 2.5 * r_to_h_time:
        return None

    # Compute neckline
    neck_run = r_armpit - l_armpit
    neck_rise = data[r_armpit] - data[l_armpit]
    neck_slope = neck_rise / neck_run
    
    # neckline value at current index
    neck_val = data[l_armpit] + (i - l_armpit) * neck_slope
    
    # Confirm pattern when price is halfway from right shoulder
    if early_find: 
        if data[i] < r_midpoint:
            return None
    else:
       
        # Price has yet to break neckline, unconfirmed
        if data[i] < neck_val:
            return None
   
    # Find beginning of pattern. Neck to left shoulder
    head_width = r_armpit - l_armpit
    pat_start = -1
    neck_start = -1
    for j in range(1, head_width):
        neck = data[l_armpit] + (l_shoulder - l_armpit - j) * neck_slope
        
        if l_shoulder - j < 0:
            return None
        
        if data[l_shoulder - j] > neck:
            pat_start = l_shoulder - j
            neck_start = neck
            break

    if pat_start == -1:
        return None

    # Pattern confirmed if here :)
    pat = HSPattern(inverted=True)  
    
    pat.l_shoulder = l_shoulder
    pat.r_shoulder = r_shoulder
    pat.l_armpit = l_armpit
    pat.r_armpit = r_armpit
    pat.head = head
    
    pat.l_shoulder_p = data[l_shoulder]
    pat.r_shoulder_p = data[r_shoulder]
    pat.l_armpit_p = data[l_armpit]
    pat.r_armpit_p = data[r_armpit]
    pat.head_p = data[head]

    pat.start_i = pat_start
    pat.break_i = i
    pat.break_p = data[i]

    pat.neck_start = neck_start
    pat.neck_end = neck_val
    pat.pattern_r2 = compute_pattern_r2(data, pat)
    
    pat.neck_slope = neck_slope
    pat.head_width = head_width
    pat.head_height = (data[l_armpit] + (head - l_armpit) * neck_slope) - data[head]
    pat.pattern_r2 = compute_pattern_r2(data, pat)
    
    #if pat.pattern_r2 < 0.0:
    #    return None

    return pat

def find_hs_patterns(data: np.array, order: int, early_find: bool = False):
    # ... (unchanged)
    assert(order >= 1)
    
    # head and shoulders top checked from/after a confirmed bottom (before right shoulder)
    # head and shoulders bottom checked from/after a confirmed top 
    
    last_is_top = False
    recent_extrema = deque(maxlen=5)
    recent_types = deque(maxlen=5) # -1 for bottoms 1 for tops

    # Lock variables to prevent finding the same pattern multiple times
    hs_lock = False
    ihs_lock = False

    ihs_patterns = [] # Inverted (bullish)
    hs_patterns = []  # Regular (bearish)
    for i in range(len(data)):

        if rw_top(data, i, order):
            recent_extrema.append(i - order)
            recent_types.append(1)
            ihs_lock = False
            last_is_top = True
        
        if rw_bottom(data, i, order):
            recent_extrema.append(i - order)
            recent_types.append(-1)
            hs_lock = False
            last_is_top = False

        if len(recent_extrema) < 5:
            continue
        
        hs_alternating = True
        ihs_alternating = True
        
        if last_is_top:
            for j in range(2, 5):
                if recent_types[j] == recent_types[j - 1]: 
                    ihs_alternating = False
            
            for j in range(1, 4):
                if recent_types[j] == recent_types[j - 1]: 
                    hs_alternating = False
            
            ihs_extrema = list(recent_extrema)[1:5]
            hs_extrema = list(recent_extrema)[0:4]
        else:
            
            for j in range(2, 5):
                if recent_types[j] == recent_types[j - 1]: 
                    hs_alternating = False
            
            for j in range(1, 4):
                if recent_types[j] == recent_types[j - 1]: 
                    ihs_alternating = False
            
            ihs_extrema = list(recent_extrema)[0:4]
            hs_extrema = list(recent_extrema)[1:5]
        
        if ihs_lock or not ihs_alternating:
            ihs_pat = None
        else:
            ihs_pat = check_ihs_pattern(ihs_extrema, data, i, early_find)

        if hs_lock or not hs_alternating:
            hs_pat = None
        else:
            hs_pat = check_hs_pattern(hs_extrema, data, i, early_find)

        if hs_pat is not None:
            hs_lock = True
            hs_patterns.append(hs_pat)
        
        if ihs_pat is not None:
            ihs_lock = True
            ihs_patterns.append(ihs_pat)

    return hs_patterns, ihs_patterns

def get_pattern_return(data: np.array, pat: HSPattern, log_prices: bool = True) -> float:
    # ... (unchanged)
    entry_price = pat.break_p
    entry_i = pat.break_i
    stop_price = pat.r_shoulder_p

    if pat.inverted:
        tp_price = pat.neck_end + pat.head_height
    else:
        tp_price = pat.neck_end - pat.head_height

    exit_price = -1
    for i in range(pat.head_width):
        if entry_i + i >= len(data):
            return np.nan

        exit_price = data[entry_i + i]
        if pat.inverted and (exit_price > tp_price or exit_price < stop_price):
            break
        
        if not pat.inverted and (exit_price < tp_price or exit_price > stop_price):
            break
    
    if pat.inverted: # Long
        if log_prices:
            return exit_price - entry_price
        else:
            return (exit_price - entry_price) / entry_price
    else: # Short
        if log_prices:
            return entry_price - exit_price
        else:
            return -1 * (exit_price - entry_price) / entry_price

def plot_hs(candle_data: pd.DataFrame, pat: HSPattern, pad: int = 2, save_path: str = 'static/hs_plot.png'):
    if pad < 0:
        pad = 0
    idx = candle_data.index
    try:
        data = candle_data.iloc[pat.start_i:pat.break_i + 1 + pad]
    except IndexError:
        print("No graph found. Check index bounds.")
        return

    idx = candle_data.index
    data = candle_data.iloc[pat.start_i:pat.break_i + 1 + pad]

    plt.style.use('dark_background')
    fig = plt.gcf()
    ax = fig.gca()

    l0 = [(idx[pat.start_i], pat.neck_start), (idx[pat.l_shoulder], pat.l_shoulder_p)]
    l1 = [(idx[pat.l_shoulder], pat.l_shoulder_p), (idx[pat.l_armpit], pat.l_armpit_p)]
    l2 = [(idx[pat.l_armpit], pat.l_armpit_p ), (idx[pat.head], pat.head_p)]
    l3 = [(idx[pat.head], pat.head_p ), (idx[pat.r_armpit], pat.r_armpit_p)]
    l4 = [(idx[pat.r_armpit], pat.r_armpit_p ), (idx[pat.r_shoulder], pat.r_shoulder_p)]
    l5 = [(idx[pat.r_shoulder], pat.r_shoulder_p ), (idx[pat.break_i], pat.neck_end)]
    neck = [(idx[pat.start_i], pat.neck_start), (idx[pat.break_i], pat.neck_end)]

    mpf.plot(data, alines=dict(alines=[l0, l1, l2, l3, l4, l5, neck], colors=['w', 'w', 'w', 'w', 'w', 'w', 'r']),
             type='candle', style='charles', ax=ax)
    x = len(data) // 2 - len(data) * 0.1
    if pat.inverted:
        y = pat.head_p + pat.head_height * 1.25
    else:
        y = pat.head_p - pat.head_height * 1.25

    # ax.text(x, y, f"{symbol} from {start_date} to {end_date} ({idx[pat.start_i].strftime('%Y-%m-%d %H:%M')} - {idx[pat.break_i].strftime('%Y-%m-%d %H:%M')})", color='white', fontsize='xx-large')

    # Save the plot to a file
    canvas = FigureCanvas(fig)
    canvas.print_png(save_path)

def fibonacci_retracement_levels(price_high, price_low):
    levels = [0.236, 0.382, 0.5, 1.0]
    return [(price_high - (price_high - price_low) * level) for level in levels]

@app.route('/')
def home():
    return render_template('indexx.html')

@app.route('/predict', methods=['POST'])
def predict():
    stock_symbol = request.form['stock_name']
    api_key = 'GZEDAVQVGRBZAV2X'
    symbol = stock_symbol
    url = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={api_key}&period=quarter'
    r = requests.get(url)
    data = r.json()
    quarterly_data = data.get('quarterlyReports',[])
    df = pd.DataFrame(quarterly_data)
    df=df.head(10)
    table_html = df.to_html(classes='table table-striped', index=False)

    start_date = '2023-04-16'
    end_date = '2024-04-15'
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    stock_data['EMA_200'] = stock_data['Close'].ewm(span=200, adjust=False).mean()
    stock_data['EMA_100'] = stock_data['Close'].ewm(span=100, adjust=False).mean()
    stock_data['EMA_50'] = stock_data['Close'].ewm(span=50, adjust=False).mean()
    plt.figure(figsize=(10, 5))
    
    plt.plot(stock_data.index, stock_data['Close'], label=stock_symbol)

    retracement_levels = fibonacci_retracement_levels(stock_data['High'].max(), stock_data['Low'].min())
    for level in retracement_levels:
        plt.axhline(y=level, color='r', linestyle='--', label=f'Fib {level:.2f}')
    plt.axhspan(retracement_levels[1], retracement_levels[2], color='gold', alpha=0.3, label='Fib Zone (0.382 - 0.5)')
    plt.legend()
    plt.title(f'{stock_symbol} Price Chart with Fibonacci Retracement Levels')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    
    bollinger_image_path = 'static/bollinger_bands.png'
    basis = stock_data['Close'].rolling(window=20).mean()
    dev = 2 * stock_data['Close'].rolling(window=20).std()
    upper = basis + dev
    lower = basis - dev
    plt.figure(figsize=(10, 5))
    plt.plot(stock_data.index, stock_data['Close'], label=stock_symbol)
    plt.plot(basis, label='Moving Average')
    plt.plot(upper, label='Upper Band', linestyle='--')
    plt.plot(lower, label='Lower Band', linestyle='--')
    plt.title(f'{stock_symbol} Price Chart with Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(bollinger_image_path)
    plt.close()

    current_price = stock_data['Close'].iloc[-1]
    buy_threshold = 0
    sell_threshold = 1
    if current_price <= buy_threshold:
        signal = "Buy"
    elif current_price >= sell_threshold:
        signal = "Sell"
    else:
        signal = "Hold"

    stock_data['Daily_Return'] = stock_data['Adj Close'].pct_change()
    autocorrelation = stock_data['Daily_Return'].autocorr()

    buy_threshold = 0.2
    sell_threshold = -0.2
    if autocorrelation > buy_threshold:
        autocorrelation_signal = "Buy"
    elif autocorrelation < sell_threshold:
        autocorrelation_signal = "Sell"
    else:
        autocorrelation_signal = "Hold"

    sma_period = 50
    stock_data['SMA'] = stock_data['Volume'].rolling(window=sma_period).mean()

    def generate_signals(data):
        signals=[]
        position=None
        for i in range(len(data)):
            if data['Volume'][i]>data['SMA'][i]:
                signals.append(1)
                position='long'
            else:
                signals.append(-1)
                position='short'
        return signals, position

    signals, position = generate_signals(stock_data)
    stock_data['Signal'] = signals

    if position == 'long':
        volume_signal = "Buy"
    elif position == 'short':
        volume_signal = "Sell"
    else:
        volume_signal = "No recommendation"

    closing_prices = stock_data['Close'].values
    fourier_transform = fft(closing_prices)
    frequencies = np.fft.fftfreq(len(fourier_transform))
    high_freq_transform = fourier_transform.copy()
    high_freq_transform[np.abs(frequencies) <= 0.1] = 0
    high_freq_series = np.fft.ifft(high_freq_transform).real
    peaks, _ = find_peaks(high_freq_series)
    high_freq_peak_dates = stock_data.index[peaks][:5]

    low_freq_transform = fourier_transform.copy()
    low_freq_transform[np.abs(frequencies) > 0.1] = 0
    low_freq_series = np.fft.ifft(low_freq_transform).real
    low_peaks, _ = find_peaks(low_freq_series)
    low_freq_peak_dates = stock_data.index[low_peaks][:5]

    high_freq_df = pd.DataFrame({'High_Frequency_Peak_Date': high_freq_peak_dates})
    low_freq_df = pd.DataFrame({'Low_Frequency_Peak_Date': low_freq_peak_dates})

    image_path = 'static/price_chart.png'
    plt.savefig(image_path)
    plt.close()
    
    # Candlestick chart with EMAs
    candlestick_image_path = 'static/candlestick_chart.png'
    plt.figure(figsize=(10, 5))
# Plot candlestick chart
# Add candlestick plotting code here using the 'data' DataFrame

# Plot EMAs
    plt.plot(stock_data.index, stock_data['Close'], label='Close', color='black')
    plt.plot(stock_data.index, stock_data['EMA_200'], label='EMA 200', color='blue')
    plt.plot(stock_data.index, stock_data['EMA_100'], label='EMA 100', color='green')
    plt.plot(stock_data.index, stock_data['EMA_50'], label='EMA 50', color='red')

# Customize the plot
    plt.title('Candlestick Chart with EMAs')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    # Save the candlestick chart image
    plt.savefig(candlestick_image_path)
    plt.close()
    data = yf.download(symbol, start_date, end_date, interval='1D')

    # Convert the index to datetime
    data.index = data.index.astype('datetime64[s]')

    # Take the logarithm of the data
    data = np.log(data)

    # Slice the 'Close' column and convert it to a NumPy array
    dat_slice = data['Close'].to_numpy()

    # Create head and shoulders plot
    hs_patterns, ihs_patterns = find_hs_patterns(dat_slice, 6, early_find=False)

    hs_df = pd.DataFrame()
    ihs_df = pd.DataFrame() 

    # Load pattern attributes into dataframes
    for i, hs in enumerate(hs_patterns):
        hs_df.loc[i, 'head_width'] = hs.head_width
        hs_df.loc[i, 'head_height'] = hs.head_height
        hs_df.loc[i, 'r2'] = hs.pattern_r2
        hs_df.loc[i, 'neck_slope'] = hs.neck_slope

        hp = int(hs.head_width)
        if hs.break_i + hp >= len(data):
            hs_df.loc[i, 'hold_return'] = np.nan
        else:
            ret = -1 * (dat_slice[hs.break_i + hp] - dat_slice[hs.break_i])
            hs_df.loc[i, 'hold_return'] = ret

        hs_df.loc[i, 'stop_return'] = get_pattern_return(dat_slice, hs)

    for i, hs in enumerate(ihs_patterns):
        ihs_df.loc[i, 'head_width'] = hs.head_width
        ihs_df.loc[i, 'head_height'] = hs.head_height
        ihs_df.loc[i, 'r2'] = hs.pattern_r2
        ihs_df.loc[i, 'neck_slope'] = hs.neck_slope

        hp = int(hs.head_width)
        if hs.break_i + hp >= len(data):
            ihs_df.loc[i, 'hold_return'] = np.nan
        else:
            ret = dat_slice[hs.break_i + hp] - dat_slice[hs.break_i]
            ihs_df.loc[i, 'hold_return'] = ret

        ihs_df.loc[i, 'stop_return'] = get_pattern_return(dat_slice, hs)

    # Plot and save the first image
    ihs_image_path = 'static/ihs_plot.png'
    if ihs_patterns:
        plot_hs(data, ihs_patterns[0], pad=50, save_path=ihs_image_path)
        plt.close()  # Close the plot after saving
    else:
        # Remove the previous image if no pattern is detected
        if os.path.exists(ihs_image_path):
            os.remove(ihs_image_path)

    # Plot and save the second image
    hs_image_path = 'static/hs_plot.png'
    if hs_patterns:
        plot_hs(data, hs_patterns[0], pad=50, save_path=hs_image_path)
        plt.close()  # Close the plot after saving
    else:
        # Remove the previous image if no pattern is detected
        if os.path.exists(hs_image_path):
            os.remove(hs_image_path)
   


    return render_template('result.html',
                           income_statement=df.to_html(), table=df.to_html(),
                           stock_name=stock_symbol, current_price=current_price,
                           signal=signal,
                           autocorrelation=autocorrelation,
                           autocorrelation_signal=autocorrelation_signal,
                           volume_signal=volume_signal,
                           high_freq_peaks=high_freq_df.to_html,
                           low_freq_peaks=low_freq_df.to_html,
                           price_chart=image_path,
                           bollinger_chart=bollinger_image_path,
                           candlestick_chart_path=candlestick_image_path, ihs_image_path=ihs_image_path,
                           hs_image_path=hs_image_path,
                           
                           )

if __name__ == '__main__':
    app.run(debug=True)

