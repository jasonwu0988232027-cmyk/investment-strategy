import yfinance as yf
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore') # 忽略一些 pandas 計算產生的警告

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
        print("資料筆數不足，無法計算指標設定 (至少需要60天)")
        return None
        
    df['ATR'] = calculate_atr(df)
    
    # 趨勢確認與價格突破指標
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['High_20'] = df['High'].rolling(window=20).max().shift(1)
    
    df['RSI'] = calculate_rsi(df)
    
    # 計算買入訊號與強勢度評分
    df['Signal_Buy'] = df['RSI'] <= 30
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
    
    atr_values = df['ATR'].values
    high_values = df['High'].values
    low_values = df['Low'].values
    
    for i in range(1, len(df)):
        today_high = high_values[i]
        today_low = low_values[i]
        today_atr = atr_values[i]
        
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
                    swing_low = np.nanmin(low_values[old_high_idx:high_idx+1])
                else:
                    swing_low = np.nanmin(low_values[:high_idx+1])
            
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

def run_backtest(df, ticker, initial_capital=100000.0):
    print(f"\n========== 開始執行 {ticker} 回測模擬 ==========")
    capital = initial_capital
    position = 0
    entry_price = 0
    entry_date = None
    trades = []
    
    dates = df.index
    open_prices = df['Open'].values
    high_prices = df['High'].values
    low_prices = df['Low'].values
    close_prices = df['Close'].values
    buy_signals = df['Signal_Buy'].values
    stop_losses = df['Stop_Loss'].values
    take_profits = df['Take_Profit'].values
    
    equity_curve = []
    
    for i in range(1, len(df)):
        current_date = dates[i]
        O, H, L, C = open_prices[i], high_prices[i], low_prices[i], close_prices[i]
        
        # 判斷是否觸發平倉 (使用昨天的 SL / TP 設定)
        if position > 0:
            sl = stop_losses[i-1]
            tp = take_profits[i-1]
            
            sold = False
            sell_price = 0
            reason = ""
            
            # 若開盤直接跳空跌破停損
            if O <= sl:
                sold = True
                sell_price = O
                reason = "Stop Loss (Gap Down)"
            # 盤中跌破停損
            elif L <= sl:
                sold = True
                sell_price = sl
                reason = "Stop Loss"
            # 若開盤直接跳空突破停利
            elif O >= tp:
                sold = True
                sell_price = O
                reason = "Take Profit (Gap Up)"
            # 盤中突破停利
            elif H >= tp:
                sold = True
                sell_price = tp
                reason = "Take Profit"
                
            if sold:
                capital = position * sell_price
                ret = (sell_price - entry_price) / entry_price
                trades.append({
                    '進場時間': entry_date.date() if hasattr(entry_date, 'date') else entry_date,
                    '出場時間': current_date.date() if hasattr(current_date, 'date') else current_date,
                    '進場價格': round(entry_price, 2),
                    '出場價格': round(sell_price, 2),
                    '報酬率(%)': round(ret * 100, 2),
                    '出場原因': reason
                })
                position = 0
        
        # 判斷是否可以進場 (昨天出現買進訊號，今天開盤買進)
        if position == 0 and buy_signals[i-1]:
            # 用今日開盤價買進 (假設市價買入，全倉操作)
            if not pd.isna(O):
                position = capital / O
                entry_price = O
                entry_date = current_date
                capital = 0 
            
        # 紀錄每日收盤淨值
        current_equity = position * C if position > 0 else capital
        equity_curve.append({
            'Date': current_date,
            'Equity': current_equity
        })
        
    # 回測結束，按照最後一天收盤價清倉
    if position > 0:
        current_equity = position * close_prices[-1]
        ret = (close_prices[-1] - entry_price) / entry_price
        trades.append({
            '進場時間': entry_date.date() if hasattr(entry_date, 'date') else entry_date,
            '出場時間': dates[-1].date() if hasattr(dates[-1], 'date') else dates[-1],
            '進場價格': round(entry_price, 2),
            '出場價格': round(close_prices[-1], 2),
            '報酬率(%)': round(ret * 100, 2),
            '出場原因': "End of Backtest"
        })
    else:
        current_equity = capital
        
    equity_df = pd.DataFrame(equity_curve).set_index('Date')
    
    # 輸出分析報告
    total_return = ((current_equity - initial_capital) / initial_capital) * 100
    trades_df = pd.DataFrame(trades)
    
    print("\n========== 回測績效報告 ==========")
    print(f"回測標的:\t {ticker}")
    print(f"初始資金:\t ${initial_capital:,.2f}")
    print(f"最終資金:\t ${current_equity:,.2f}")
    print(f"總報酬率:\t {total_return:.2f}%")
    
    if not trades_df.empty:
        win_trades = trades_df[trades_df['報酬率(%)'] > 0]
        loss_trades = trades_df[trades_df['報酬率(%)'] <= 0]
        win_rate = len(win_trades) / len(trades_df) * 100
        
        print(f"總交易次數:\t {len(trades_df)}")
        print(f"勝率:\t\t {win_rate:.2f}% ({len(win_trades)} 勝 / {len(loss_trades)} 敗)")
        print(f"平均每筆報酬:\t {trades_df['報酬率(%)'].mean():.2f}%")
        if len(win_trades) > 0:
            print(f"平均獲利報酬:\t {win_trades['報酬率(%)'].mean():.2f}%")
        if len(loss_trades) > 0:
            print(f"平均虧損報酬:\t {loss_trades['報酬率(%)'].mean():.2f}%")
            
        equity_df['High_Water_Mark'] = equity_df['Equity'].cummax()
        equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['High_Water_Mark']) / equity_df['High_Water_Mark']
        max_drawdown = equity_df['Drawdown'].min() * 100
        print(f"最大回撤 (MDD):\t {max_drawdown:.2f}%")
        
        print("\n--- 交易明細 (前 5 筆) ---")
        print(trades_df.head(5).to_string(index=False))
        if len(trades_df) > 10:
            print("...\n--- 交易明細 (最後 5 筆) ---")
            print(trades_df.tail(5).to_string(index=False))
        elif len(trades_df) > 5:
            print("...\n")
            print(trades_df.tail(len(trades_df)-5).to_string(index=False))
            
        # 將所有交易紀錄輸出至 CSV
        csv_filename = f"{ticker}_trades.csv"
        trades_df.to_csv(csv_filename, index=False)
        print(f"\n💡 完整交易明細已匯出至 {csv_filename}")
        
    else:
        print("沒有發生任何交易 (未觸發進場訊號)。")
        
    # 歷史權益曲線圖表
    try:
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
        plt.rcParams['axes.unicode_minus'] = False 
        
        plt.figure(figsize=(12, 6))
        plt.plot(equity_df.index, equity_df['Equity'], label='策略累積淨值 (Equity)', color='b')
        plt.fill_between(equity_df.index, equity_df['Equity'], initial_capital, where=(equity_df['Equity'] > initial_capital), color='g', alpha=0.3)
        plt.fill_between(equity_df.index, equity_df['Equity'], initial_capital, where=(equity_df['Equity'] <= initial_capital), color='r', alpha=0.3)
        plt.axhline(initial_capital, color='black', linestyle='--', label='初始資金')
        plt.title(f'【{ticker}】策略回測權益曲線', fontsize=16)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('淨值', fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        png_filename = f"{ticker}_equity_curve.png"
        plt.savefig(png_filename)
        print(f"💡 權益曲線圖表已儲存為 {png_filename}")
    except ImportError:
        pass

def main():
    print("====== 股票策略單一標的回測系統 ======")
    ticker = input("請輸入要回測的股票代碼 (例如 2330.TW, 2603.TW): ").strip()
    if not ticker:
        print("未輸入代碼，預設使用 2330.TW")
        ticker = "2330.TW"
        
    print(f"\n[{ticker}] 正在下載歷史資料...")
    df = yf.download(ticker, period="1y", progress=False)
    
    # 處理 yfinance 返回格式
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, level=1, axis=1)
        except:
            if len(df.columns.levels) > 1 and ticker in df.columns.levels[0]:
                df = df.xs(ticker, level=0, axis=1)
            else:
                df.columns = df.columns.droplevel(1)
            
    df.dropna(how='all', inplace=True)
    if 'Close' not in df.columns or df.empty:
        print("發生錯誤：下載的資料異常或不包含收盤價。")
        return
        
    print("資料下載完成，計算策略指標與買賣訊號中...")
    df_strategy = apply_strategy(df)
    
    if df_strategy is None:
        return
        
    # 執行回測引擎
    run_backtest(df_strategy, ticker)

if __name__ == "__main__":
    main()
