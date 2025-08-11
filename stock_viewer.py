import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class StockViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Viewer")
        self.root.geometry("800x600")

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
            df = yf.download(ticker, start='2020-01-01', end='2023-01-01')
            if df.empty:
                messagebox.showerror("Error", f"No data found for ticker: {ticker}")
                return
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        for widget in self.output_frame.winfo_children():
            widget.destroy()
        if self.display_var.get() == "Graph":
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(df.index, df['Close'], label='Close Price')
            ax.set_title(f"{ticker} Close Price")
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            canvas = FigureCanvasTkAgg(fig, master=self.output_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            tree = ttk.Treeview(self.output_frame, columns=list(df.columns), show='headings')
            for col in df.columns:
                tree.heading(col, text=col)
                tree.column(col, width=100)
            for idx, row in df.head(50).iterrows():
                tree.insert('', 'end', values=list(row))
            tree.pack(fill=tk.BOTH, expand=True)
            scrollbar = ttk.Scrollbar(self.output_frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side='right', fill='y')

if __name__ == "__main__":
    root = tk.Tk()
    app = StockViewerApp(root)
    root.mainloop()
