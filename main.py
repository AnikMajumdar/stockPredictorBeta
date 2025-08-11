import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ticker = 'AAPL'
df = yf.download(ticker, start='2020-01-01', end='2023-01-01')
print(df)
df.Close.plot(figsize=(14, 7))
plt.show()
    
scaler = StandardScaler()
predicted_prices = df['Close'] * 0.98  # Example: 2% below actual prices

# Plot both actual and predicted prices
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Close'], label='Actual Price')
plt.plot(df.index, predicted_prices, label='Predicted Price')
plt.legend()
plt.title('Actual vs Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

scaler - StandardScaler()
scaler.fit(df[['Close']])
df['Close'] = scaler.fit_transform(df[['Close']])

seq_length = 30
data = []
for i in range(len(df) - seq_length):
    data.append(df.Close[i:i + seq_length])

data = np.array(data)
train_size = int(len(data) * 0.8)
x_train = torch.from_numpy(data[:train_size, :-1, :]).type(torch.Tensor).to(device)
y_train = torch.from_numpy(data[:train_size, -1, :]).type(torch.Tensor).to(device)
X_test = torch.from_numpy(data[:train_size, :-1, :]).type(torch.Tensor).to(device)
y_test = torch.from_numpy(data[:train_size, -1, :]).type(torch.Tensor).to(device)

class PredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PredictionModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device = device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device = device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

model = PredictionModel(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for i in range(num_epochs):
    y_train_pred = model(x_train)
    loss = criterion(y_train_pred, y_train)
    
    if i% 24 == 0:
        print (i, loss.item())

optimizer.zero_grad()
loss.backward()
optimizer.step()

model.eval()
y_test_pred = model(X_test)

y_train_pred = scaler.inverse_transform(y_train_pred.detach().cpu().numpy())
y_train = scaler.inverse_transform(y_train.detach().cpu().numpy())
y_test_pred = scaler.inverse_transform(y_test.detach().cpu().numpy())
y_test = scaler.inverse_transform(y_test.detach().cpu().numpy())

print('y_test_pred shape:', y_test_pred.shape)
print('y_test shape:', y_test.shape)
print('First 10 predicted prices:', y_test_pred[:10])
print('First 10 actual prices:', y_test[:10])

train_rmse = root_mean_squared_error(y_train[:, 0], y_train_pred[:, 0])
test_rmse = root_mean_squared_error(y_test[:, 0], y_test_pred[:, 0])

print(f'Train RMSE: {train_rmse:.2f}')
print(f'Test RMSE: {test_rmse:.2f}')


# Create a figure with two subplots: one for prices, one for error
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

# Actual vs Predicted Prices
ax1.plot(df.iloc[-len(y_test):].index, y_test, color='blue', label='Actual Price')
ax1.plot(df.iloc[-len(y_test):].index, y_test_pred, color='green', label='Predicted Price')
ax1.legend()
ax1.set_title(f"{ticker} Stock Price Prediction")
ax1.set_xlabel('Date')
ax1.set_ylabel('Price')

# Prediction Error
ax2.plot(df.iloc[-len(y_test):].index, abs(y_test - y_test_pred), color='red', label='Prediction Error')
ax2.legend()
ax2.set_title("Prediction Error")
ax2.set_xlabel('Date')
ax2.set_ylabel('Error')

# Add RMSE table below the error graph
cell_text = [[f'{train_rmse:.2f}'], [f'{test_rmse:.2f}']]
row_labels = ['Train RMSE', 'Test RMSE']
table = ax2.table(cellText=cell_text, rowLabels=row_labels, colLabels=['RMSE'], loc='bottom', cellLoc='center')
table.scale(1, 1.5)
plt.subplots_adjust(bottom=0.2)

plt.tight_layout()
plt.show()