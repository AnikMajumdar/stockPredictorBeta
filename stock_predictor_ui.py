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
        return closes
    X = closes[:-1].reshape(-1, 1)
    y = closes[1:]
    model = LinearRegression()
    model.fit(X, y)
    # Predict for all except the first day
    predicted = model.predict(closes[:-1].reshape(-1, 1))
    # Pad the first value to align with actuals
    predicted = np.insert(predicted, 0, closes[0])
    return predicted

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

        # Display option
        self.display_var = tk.StringVar(value="Graph")
        self.graph_radio = ttk.Radiobutton(root, text="Graph", variable=self.display_var, value="Graph")
        self.table_radio = ttk.Radiobutton(root, text="Table", variable=self.display_var, value="Table")
        self.graph_radio.pack()
        self.table_radio.pack()

        # Submit button
        self.submit_btn = ttk.Button(root, text="Show", command=self.show_data)
        self.submit_btn.pack(pady=10)

        # Output frame
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
        predicted = predict_prices(df)
        actual = df['Close'].values
        rmse = calculate_rmse(actual, predicted)
        if self.display_var.get() == "Graph":
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df.index, actual, label='Actual Price', color='blue')
            ax.plot(df.index, predicted, label='Predicted Price', color='green')
            ax.set_title(f"{ticker} Actual vs Predicted Prices\nRMSE: {rmse:.2f}")
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            # Add RMSE table below the graph
            cell_text = [[f'{rmse:.2f}']]
            row_labels = ['RMSE']
            table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=['Error'], loc='bottom', cellLoc='center')
            table.scale(1, 1.5)
            plt.subplots_adjust(bottom=0.2)
            canvas = FigureCanvasTkAgg(fig, master=self.output_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            # Table view: show actual, predicted, and error
            table_df = pd.DataFrame({
                'Date': df.index.strftime('%Y-%m-%d'),
                'Actual': [f'{v:.2f}' for v in actual],
                'Predicted': [f'{v:.2f}' for v in predicted],
                'Error': [f'{v:.2f}' for v in np.abs(actual - predicted)]
            })
            if table_df.empty:
                print("DEBUG: Table DataFrame is empty.")
                error_label = ttk.Label(self.output_frame, text="No data available to display in table.", font=("Arial", 12, "bold"))
                error_label.pack(pady=10)
            else:
                print(f"DEBUG: Table DataFrame shape: {table_df.shape}")
                # Limit to 100 rows for performance
                display_df = table_df.head(100)
                tree = ttk.Treeview(self.output_frame, columns=list(display_df.columns), show='headings', height=20)
                style = ttk.Style()
                style.configure("Treeview.Heading", font=("Arial", 11, "bold"))
                style.configure("Treeview", font=("Arial", 10))
                for col in display_df.columns:
                    tree.heading(col, text=col)
                    tree.column(col, width=120, anchor='center')
                for _, row in display_df.iterrows():
                    tree.insert('', 'end', values=list(row))
                tree.pack(fill=tk.BOTH, expand=True)
                scrollbar = ttk.Scrollbar(self.output_frame, orient="vertical", command=tree.yview)
                tree.configure(yscrollcommand=scrollbar.set)
                scrollbar.pack(side='right', fill='y')
                # Show RMSE below the table
                rmse_label = ttk.Label(self.output_frame, text=f"RMSE: {rmse:.2f}", font=("Arial", 12, "bold"))
                rmse_label.pack(pady=10)
                messagebox.showinfo("Table Created", f"Table successfully created with {display_df.shape[0]} rows.")

if __name__ == "__main__":
    root = tk.Tk()
    app = StockPredictorUI(root)
    root.mainloop()
