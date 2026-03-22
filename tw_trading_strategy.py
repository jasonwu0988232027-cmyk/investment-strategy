import yfinance as yf
import pandas as pd
import numpy as np
import time
import os
import requests
import urllib3
import datetime

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 定義標的池 (Universe): 大型權值股、金融股、航運股
UNIVERSE = [
    '2330.TW', '2317.TW', '2454.TW', '2308.TW', '2382.TW', # 權值股
    '2881.TW', '2882.TW', '2883.TW', '2884.TW', '2885.TW', # 金融股
    '2886.TW', '2887.TW', '2890.TW', '2891.TW', '2892.TW',
    '2603.TW', '2609.TW', '2615.TW', # 航運股
    '2606.TW', '2610.TW', '2618.TW'
]

def fetch_data(tickers):
    all_data = []
    chunk_size = 20
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        print(f"Downloading batch {i//chunk_size + 1}/{len(tickers)//chunk_size + 1}: {chunk}")
        
        df = yf.download(chunk, period="1y", group_by="ticker", progress=False)
        all_data.append(df)
        time.sleep(1)
        
    processed_dfs = []
    for chunk, df in zip([tickers[i:i+chunk_size] for i in range(0, len(tickers), chunk_size)], all_data):
        if len(chunk) == 1:
            ticker = chunk[0]
            # Convert SingleIndex to MultiIndex to match batched requests
            if not isinstance(df.columns, pd.MultiIndex):
                df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
        processed_dfs.append(df)
        
    final_df = pd.concat(processed_dfs, axis=1)
    
    # 自動刪除完全沒有數據的欄位
    final_df.dropna(how='all', axis=1, inplace=True)
    return final_df

def fetch_top100_from_twse_tpex():
    print("Fetching whole market data from TWSE and TPEX...")
    all_stocks = []
    
    try:
        url_twse = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        resp = requests.get(url_twse, verify=False, timeout=10)
        data_twse = resp.json()
        for d in data_twse:
            code = d.get('Code', '')
            if len(code) == 4 and code.isdigit():
                try: tv = float(d.get('TradeValue', 0))
                except: tv = 0
                all_stocks.append({'Ticker': code + '.TW', 'TradeValue': tv})
    except Exception as e:
        print(f"Error fetching TWSE: {e}")
        
    try:
        url_tpex = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes"
        resp = requests.get(url_tpex, verify=False, timeout=10)
        data_tpex = resp.json()
        for d in data_tpex:
            code = d.get('SecuritiesCompanyCode', '')
            if len(code) == 4 and code.isdigit():
                try: tv = float(d.get('TransactionAmount', 0))
                except: tv = 0
                all_stocks.append({'Ticker': code + '.TWO', 'TradeValue': tv})
    except Exception as e:
        print(f"Error fetching TPEX: {e}")
        
    df = pd.DataFrame(all_stocks)
    if not df.empty:
        df = df.sort_values(by='TradeValue', ascending=False)
        top_100 = df.head(100)['Ticker'].tolist()
        return top_100
    return []

def fetch_historical_data_finmind(tickers):
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365)
    
    all_dfs = []
    print(f"Fetching data from FinMind for {len(tickers)} tickers...")
    
    for i, ticker in enumerate(tickers):
        stock_id = ticker.split('.')[0]
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            "dataset": "TaiwanStockPrice",
            "data_id": stock_id,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d")
        }
        try:
            resp = requests.get(url, params=params, verify=False, timeout=10)
            data = resp.json()
            if data.get('msg') == 'success' and len(data.get('data', [])) > 0:
                df = pd.DataFrame(data['data'])
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'Trading_Volume': 'Volume'})
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
                all_dfs.append(df)
            else:
                pass # print(f"FinMind missing data: {ticker}")
        except Exception as e:
            print(f"Error FinMind {ticker}: {e}")
            
        time.sleep(0.5)
        if i > 0 and i % 20 == 0:
            print(f"  Processed {i}/{len(tickers)} stocks...")
            
    if all_dfs:
        final_df = pd.concat(all_dfs, axis=1)
        return final_df
    return pd.DataFrame()

def calculate_atr(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def apply_strategy(df_ticker):
    df = df_ticker.copy()
    if df.empty or len(df) < 60:
        return None
        
    df['ATR'] = calculate_atr(df)
    
    # 趨勢確認與價格突破指標
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['High_20'] = df['High'].rolling(window=20).max().shift(1)
    
    df['RSI'] = calculate_rsi(df)
    
    # 計算買入訊號與強勢度評分
    conditions = [
        df['RSI'] <= 30,
        (df['RSI'] > 30) & (df['RSI'] <= 50)
    ]
    choices = ['推薦入場', '觀望考慮入場']
    df['Entry_Status'] = np.select(conditions, choices, default='')
    df['Score'] = (df['Close'] / df['MA60']) - 1
    
    stop_loss = np.full(len(df), np.nan)
    take_profit = np.full(len(df), np.nan)
    fib_236 = np.full(len(df), np.nan)
    fib_382 = np.full(len(df), np.nan)
    fib_500 = np.full(len(df), np.nan)
    fib_618 = np.full(len(df), np.nan)
    fib_786 = np.full(len(df), np.nan)
    
    global_high = -np.inf
    high_idx = 0
    swing_low = np.inf
    
    for i in range(1, len(df)):
        today_high = df['High'].iloc[i]
        today_low = df['Low'].iloc[i]
        today_atr = df['ATR'].iloc[i]
        
        # 1. Trailing Stop Loss
        if not np.isnan(today_atr) and not np.isnan(today_high):
            current_sl = today_high - 2 * today_atr
            if np.isnan(stop_loss[i-1]):
                stop_loss[i] = current_sl
            else:
                stop_loss[i] = max(stop_loss[i-1], current_sl)
        else:
            stop_loss[i] = stop_loss[i-1]
            
        # 2. Dynamic Fibonacci & Take Profit
        if not np.isnan(today_high):
            if today_high > global_high:
                # 新的高點出現
                old_high_idx = high_idx
                global_high = today_high
                high_idx = i
                
                # 在舊高點和新高點之間尋找最低點
                if old_high_idx < high_idx and old_high_idx > 0:
                    swing_low = df['Low'].iloc[old_high_idx:high_idx+1].min()
                else:
                    swing_low = df['Low'].iloc[:high_idx+1].min()
            
            # Fibonacci nodes
            if global_high != -np.inf and swing_low != np.inf:
                diff = global_high - swing_low
                fib_236[i] = global_high - 0.236 * diff
                fib_382[i] = global_high - 0.382 * diff
                fib_500[i] = global_high - 0.500 * diff
                fib_618[i] = global_high - 0.618 * diff
                fib_786[i] = global_high - 0.786 * diff
                
                take_profit[i] = global_high
    
    df['Stop_Loss'] = stop_loss
    df['Take_Profit'] = take_profit
    df['Fib_0.236'] = fib_236
    df['Fib_0.382'] = fib_382
    df['Fib_0.500'] = fib_500
    df['Fib_0.618'] = fib_618
    df['Fib_0.786'] = fib_786
    
    return df

def main():
    # 1. 取得全台股前100名
    top_100_tickers = fetch_top100_from_twse_tpex()
    
    if not top_100_tickers:
        print("無法從 TWSE/TPEX 取得資料，退回使用預設 UNIVERSE。")
        top_100_tickers = UNIVERSE
    else:
        print(f"--- 成功從證交所/櫃買中心取得全市場前 100 名熱門股 ---")
        
    print(f"Top 100 Tickers: {top_100_tickers[:10]} ...")
    
    # 2. 獲取歷史資料
    # 可以切換資料來源：'yfinance' 或 'finmind'
    DATA_SOURCE = 'yfinance'  # 使用者可自行更改此處
    
    if DATA_SOURCE == 'yfinance':
        print("\nFetching historical data from Yahoo Finance...")
        df_all = fetch_data(top_100_tickers)
    elif DATA_SOURCE == 'finmind':
        print("\nFetching historical data from FinMind...")
        df_all = fetch_historical_data_finmind(top_100_tickers)
    else:
        df_all = pd.DataFrame()
        
    results = {}
    
    if not df_all.empty and isinstance(df_all.columns, pd.MultiIndex):
        if len(df_all.columns.levels) == 2:
            if 'Close' in df_all.columns.levels[0]:
                is_price_first = True
            else:
                is_price_first = False
                
            for ticker in top_100_tickers:
                try:
                    if is_price_first:
                        if ticker in df_all.columns.levels[1]:
                            df_ticker = df_all.xs(ticker, level=1, axis=1).copy()
                        else: continue
                    else:
                        if ticker in df_all.columns.levels[0]:
                            df_ticker = df_all[ticker].copy()
                        else: continue
                        
                    df_ticker.dropna(how='all', inplace=True)
                    res = apply_strategy(df_ticker)
                    if res is not None:
                        results[ticker] = res.iloc[-1].to_dict()
                except Exception as e:
                    print(f"Error processing {ticker}: {e}")
    else:
        print("Data fetch failed or unexpected columns format.")
        
    summary_df = pd.DataFrame(results).T
    if not summary_df.empty:
        # 依據強勢度評分 (Score) 進行降序排序
        if 'Score' in summary_df.columns:
            summary_df.sort_values(by='Score', ascending=False, inplace=True)
            
        # 選取部分欄位顯示
        cols_to_show = ['Close', 'Entry_Status', 'RSI', 'Score', 'Stop_Loss', 'Take_Profit', 'Fib_0.618']
        # 只保留存在的欄位
        cols_to_show = [c for c in cols_to_show if c in summary_df.columns]
        
        print("\n--- Latest Trading Signals (Sorted by Strength Score) ---")
        print(summary_df[cols_to_show])
        
        recommend_candidates = summary_df[summary_df['Entry_Status'] == '推薦入場']
        watch_candidates = summary_df[summary_df['Entry_Status'] == '觀望考慮入場']
        
        if not recommend_candidates.empty:
            print("\n💡 今日【推薦入場】（RSI <= 30）可選購之股票：")
            print(", ".join(list(recommend_candidates.index)))
        else:
            print("\n💡 今日無符合【推薦入場】（RSI <= 30）之股票。")
            
        if not watch_candidates.empty:
            print("\n👀 今日【觀望考慮入場】（30 < RSI <= 50）之股票：")
            print(", ".join(list(watch_candidates.index)))
        else:
            print("\n👀 今日無符合【觀望考慮入場】（30 < RSI <= 50）之股票。")
        
        output_file = "trading_signals.csv"
        summary_df.to_csv(output_file)
        print(f"\nSaved to {output_file} successfully.")
    else:
        print("No data processed.")

if __name__ == "__main__":
    main()
