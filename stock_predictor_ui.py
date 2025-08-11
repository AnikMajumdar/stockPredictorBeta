import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Dummy prediction function for demonstration
# Replace with your actual model prediction logic


from sklearn.linear_model import LinearRegression

def predict_prices(df):
    # Use previous day's close to predict today's close
    closes = df['Close'].values
    if len(closes) < 2:
        return closes, []
    X = closes[:-1].reshape(-1, 1)
    y = closes[1:]
    model = LinearRegression()
    model.fit(X, y)
    # Predict for all except the first day
    predicted = model.predict(closes[:-1].reshape(-1, 1))
    predicted = np.insert(predicted, 0, closes[0])
    # Forecast next 7 days
    future_preds = []
    last_close = closes[-1]
    for _ in range(7):
        # Ensure last_close is a scalar and input is 2D
        next_pred = model.predict(np.array(last_close).reshape(1, 1))[0]
        future_preds.append(next_pred)
        last_close = next_pred
    return predicted, future_preds

def calculate_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

class StockPredictorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Predictor UI")
        self.root.geometry("900x700")


        # Top 10 stocks reference
        top_stocks = [
            'AAPL - Apple',
            'MSFT - Microsoft',
            'GOOGL - Alphabet',
            'AMZN - Amazon',
            'NVDA - Nvidia',
            'META - Meta Platforms',
            'TSLA - Tesla',
            'BRK-B - Berkshire Hathaway',
            'JPM - JPMorgan Chase',
            'V - Visa'
        ]
        self.top_label = ttk.Label(root, text="Top 10 US Stocks (Ticker - Name):", font=("Arial", 11, "bold"))
        self.top_label.pack(pady=(10, 0))
        self.top_list = tk.Listbox(root, height=10, font=("Arial", 10))
        for stock in top_stocks:
            self.top_list.insert(tk.END, stock)
        self.top_list.configure(state='disabled')
        self.top_list.pack(pady=(0, 10))

        # Ticker input
        self.ticker_label = ttk.Label(root, text="Enter Stock Ticker:")
        self.ticker_label.pack(pady=10)
        self.ticker_entry = ttk.Entry(root)
        self.ticker_entry.pack(pady=5)

        # Only graph will be shown; no user selection
        self.submit_btn = ttk.Button(root, text="Show Graph", command=self.show_data)
        self.submit_btn.pack(pady=10)
        self.output_frame = ttk.Frame(root)
        self.output_frame.pack(fill=tk.BOTH, expand=True)

    def show_data(self):
        ticker = self.ticker_entry.get().strip()
        if not ticker:
            messagebox.showerror("Error", "Please enter a stock ticker.")
            return
        try:
            df = yf.download(ticker, start='2025-01-01', end='2025-06-30')
            if df.empty:
                messagebox.showerror("Error", f"No data found for ticker: {ticker}")
                return
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        for widget in self.output_frame.winfo_children():
            widget.destroy()
        predicted, future_preds = predict_prices(df)
        actual = df['Close'].values
        # Ensure predicted and actual arrays are the same length
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
        rmse = calculate_rmse(actual, predicted)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index[:min_len], actual, label='Actual Price', color='blue')
        ax.plot(df.index[:min_len], predicted, label='Predicted Price', color='green')
        # Future forecast
        if len(future_preds) > 0:
            last_date = df.index[-1]
            future_dates = pd.date_range(last_date, periods=8, freq='B')[1:]
            ax.plot(future_dates, future_preds, label='7-Day Forecast', color='orange', linestyle='dashed', marker='o')
            # Add vertical line to distinguish future
            ax.axvline(x=last_date, color='red', linestyle='--', linewidth=2, label='Forecast Start')
        ax.set_title(f"{ticker} Actual, Predicted & 7-Day Forecast\nRMSE: {rmse:.2f}")
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        # RMSE is now only shown in the graph title
        canvas = FigureCanvasTkAgg(fig, master=self.output_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = StockPredictorUI(root)
    root.mainloop()
