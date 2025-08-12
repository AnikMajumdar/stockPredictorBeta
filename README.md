# Stock Predictor UI

This project provides desktop applications for viewing and predicting stock prices using Python and Tkinter. The main application shows actual vs predicted stock prices with a 7-day forecast, while additional utilities are available for stock viewing and advanced LSTM modeling.

## How to Run

1. **Install dependencies** (if not already installed):
   ```powershell
   C:\stockPredictorBeta\venv\Scripts\pip.exe install yfinance pandas numpy scikit-learn matplotlib torch
   ```

2. **Run the main Stock Predictor UI:**
   ```powershell
   C:\stockPredictorBeta\venv\Scripts\python.exe C:\stockPredictorBeta\stock_predictor_ui.py
   ```

3. **Usage:**
   - The application displays a list of top 10 US stocks for reference
   - Enter a stock ticker (e.g., `AAPL`, `MSFT`, etc.)
   - Click "Show Graph" to display actual vs predicted prices with 7-day forecast
   - The graph shows historical data from Jan 2025 - Jun 2025 plus a 7-day forecast

## Available Applications

### 1. Stock Predictor UI (`stock_predictor_ui.py`) - Main Application
- **Purpose**: Shows actual vs predicted stock prices with 7-day forecast
- **Features**: 
  - Top 10 US stocks reference list
  - Linear regression prediction model
  - 7-day future price forecast
  - RMSE error calculation displayed in graph title
- **Date Range**: Jan 2025 - Jun 2025
- **Run Command**: `C:\stockPredictorBeta\venv\Scripts\python.exe C:\stockPredictorBeta\stock_predictor_ui.py`

### 2. Stock Viewer (`stock_viewer.py`) - Simple Viewer
- **Purpose**: Basic stock price viewing without predictions
- **Features**: Graph or table view of historical stock data
- **Date Range**: 2020-2023
- **Run Command**: `C:\stockPredictorBeta\venv\Scripts\python.exe C:\stockPredictorBeta\stock_viewer.py`

### 3. Advanced LSTM Model (`main.py`) - Research/Development
- **Purpose**: Advanced neural network model using PyTorch LSTM
- **Features**: 
  - LSTM neural network for stock prediction
  - GPU support (CUDA if available)
  - Train/test split with RMSE evaluation
  - Detailed error analysis and visualization
- **Note**: This is a script, not a GUI application
- **Run Command**: `C:\stockPredictorBeta\venv\Scripts\python.exe C:\stockPredictorBeta\main.py`

## Model Logic & Analysis

### Stock Predictor UI Model
- **Algorithm**: Linear Regression (`sklearn.linear_model.LinearRegression`)
- **Input**: Previous day's closing price
- **Output**: Next day's closing price prediction + 7-day forecast
- **Evaluation**: RMSE (Root Mean Squared Error)

### LSTM Model (main.py)
- **Algorithm**: Long Short-Term Memory neural network
- **Framework**: PyTorch with CUDA support
- **Sequence Length**: 30 days
- **Architecture**: 2-layer LSTM with 32 hidden units
- **Training**: 100 epochs with Adam optimizer

### Limitations
- Linear regression model is basic and may not capture complex market dynamics
- LSTM model requires significant computational resources and training time
- Stock prediction is inherently uncertain due to market volatility
- Models are for educational/demonstration purposes

## File Structure

- `stock_predictor_ui.py`: Main GUI application with linear regression predictions
- `stock_viewer.py`: Simple stock data viewer (no predictions)
- `main.py`: Advanced LSTM model for research/development
- `README.md`: This documentation file
- `venv/`: Python virtual environment with dependencies
