# Stock Predictor UI

This project provides a simple desktop application for viewing and predicting stock prices using Python and Tkinter. Users can enter a stock ticker, select a date range (currently set to Jan 2025 - Jun 2025), and choose to view either a graph or a table of actual and predicted prices, along with RMSE (error).

## How to Run

1. **Install dependencies** (in your virtual environment):
   ```powershell
   C:/pytorchStocks/.venv/Scripts/python.exe -m pip install yfinance pandas numpy scikit-learn matplotlib
   ```

2. **Run the UI application:**
   ```powershell
   C:/pytorchStocks/.venv/Scripts/python.exe c:/pytorchStocks/stock_predictor_ui.py
   ```

3. **Usage:**
   - Enter a stock ticker (e.g., `AAPL`, `MSFT`, etc.).
   - Choose between "Graph" or "Table" view.
   - Click "Show" to display results for the selected stock and date range.

## Model Logic & Analysis

### Model Overview

The current model uses linear regression for demonstration purposes. It uses the following logic:
- Downloads historical stock data using `yfinance`.
- Uses previous day's closing price to predict the next day's closing price with a linear regression model (`sklearn.linear_model.LinearRegression`).
- Calculates RMSE (Root Mean Squared Error) between actual and predicted prices.

### How It Works

1. **Data Download:**
   - The app fetches daily closing prices for the selected ticker and date range.
2. **Prediction:**
   - The model fits a linear regression using previous day's closing price to predict the next day's closing price.
   - Predicted prices are generated for the selected date range.
3. **Evaluation:**
   - RMSE is calculated to measure the average error between actual and predicted prices.

### Accuracy & Limitations

- **Accuracy:**
   - The current model uses a simple linear regression, which is a basic approach and may not capture complex market dynamics.
   - RMSE values provide a measure of error, but the model is not tuned for high accuracy.
- **Limitations:**
   - Linear regression is limited in its ability to model stock price movements, which are influenced by many factors.
   - For actual predictions, you should train more advanced models (e.g., LSTM, ARIMA) on historical data and validate their performance.

## File Structure

- `stock_predictor_ui.py`: Main UI application.
- `main.py`: Original script for data analysis and plotting.
- `README.md`: This documentation file.
