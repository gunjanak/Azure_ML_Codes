import pandas as pd
from datetime import datetime, timedelta

# 4. Split into train/test
from sklearn.model_selection import train_test_split

def custom_business_week_mean(values):
    # Filter out Saturdays
    working_days = values[values.index.dayofweek != 5]
    return working_days.mean()

#function to read stock data from Nepalipaisa.com api
def stock_dataFrame(stock_symbol,start_date='2020-01-01',weekly=False):
  """
  input : stock_symbol
            start_data set default at '2020-01-01'
            weekly set default at False
  output : dataframe of daily or weekly transactions
  """
  #print(end_date)
  today = datetime.today()
  # Calculate yesterday's date
  yesterday = today - timedelta(days=1)

  # Format yesterday's date
  formatted_yesterday = yesterday.strftime('%Y-%-m-%-d')
  print(formatted_yesterday)


  path = f'https://www.nepalipaisa.com/api/GetStockHistory?stockSymbol={stock_symbol}&fromDate={start_date}&toDate={formatted_yesterday}&pageNo=1&itemsPerPage=10000&pagePerDisplay=5&_=1686723457806'
  df = pd.read_json(path)
  theList = df['result'][0]
  df = pd.DataFrame(theList)
  #reversing the dataframe
  df = df[::-1]

  #removing 00:00:00 time
  #print(type(df['tradeDate'][0]))
  df['Date'] = pd.to_datetime(df['tradeDateString'])

  #put date as index and remove redundant date columns
  df.set_index('Date', inplace=True)
  columns_to_remove = ['tradeDate', 'tradeDateString','sn']
  df = df.drop(columns=columns_to_remove)

  new_column_names = {'maxPrice': 'High', 'minPrice': 'Low', 'closingPrice': 'Close','volume':'Volume','previousClosing':"Open"}
  df = df.rename(columns=new_column_names)

  if(weekly == True):
     weekly_df = df.resample('W').apply(custom_business_week_mean)
     df = weekly_df


  return df

import numpy as np
import pandas as pd

def create_sequences(df, window_size=5):
    """
    Create input-output sequences for time series forecasting.

    Parameters:
    - df: pandas DataFrame with 'Close' column
    - window_size: number of days to use as input (default 5)

    Returns:
    - X: numpy array of shape (n_samples, window_size) containing input sequences
    - y: numpy array of shape (n_samples,) containing target prices
    """
    close_prices = df['Close'].values
    X = []
    y = []

    # Create sliding windows
    for i in range(len(close_prices) - window_size):
        # Get the window of features
        window = close_prices[i:i+window_size]
        X.append(window)

        # Get the target (next day's close)
        target = close_prices[i+window_size]
        y.append(target)

    return np.array(X), np.array(y)

from sklearn.preprocessing import StandardScaler
import numpy as np

def normalize_data(X, y):
    """
    Normalize input sequences and target values using StandardScaler.

    Parameters:
    - X: Input sequences (n_samples, window_size)
    - y: Target values (n_samples,)

    Returns:
    - X_scaled: Normalized input sequences
    - y_scaled: Normalized target values
    - scaler: Fitted scaler object for inverse transformation
    """
    # Reshape X to 2D (n_samples * window_size, 1) for scaling
    X_reshaped = X.reshape(-1, 1)

    # Initialize and fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)

    # Reshape back to original shape
    X_scaled = X_scaled.reshape(X.shape)

    # Scale target values using the same scaler
    y_scaled = scaler.transform(y.reshape(-1, 1)).flatten()

    return X_scaled, y_scaled, scaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

    
class GRUPredictor(nn.Module):
    
    def __init__(self,input_size,hidden_size,output_size):
        super(GRUPredictor,self).__init__()
        self.gru = nn.GRU(input_size,hidden_size,batch_first=True)
        self.fc = nn.Linear(hidden_size,output_size)
        
    def forward(self,x):
        _,h_n = self.gru(x)
        return self.fc(h_n.squeeze(0))
    
# Training function with MAPE tracking
def train_model(X_train, y_train, X_test, y_test, scaler, epochs=100, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = GRUPredictor(1,32,1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create DataLoader
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    # Track metrics
    train_mape_history = []
    test_mape_history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Calculate MAPE
        model.eval()
        with torch.no_grad():
            # Training MAPE
            train_pred = model(X_train.to(device))
            train_pred = scaler.inverse_transform(train_pred.cpu().numpy())
            train_true = scaler.inverse_transform(y_train.unsqueeze(1).cpu().numpy())
            train_mape = mean_absolute_percentage_error(train_true, train_pred)

            # Test MAPE
            test_pred = model(X_test.to(device))
            test_pred = scaler.inverse_transform(test_pred.cpu().numpy())
            test_true = scaler.inverse_transform(y_test.unsqueeze(1).cpu().numpy())
            test_mape = mean_absolute_percentage_error(test_true, test_pred)

        train_mape_history.append(train_mape)
        test_mape_history.append(test_mape)

        print(f'Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f} | '
              f'Train MAPE: {train_mape:.2f}% | Test MAPE: {test_mape:.2f}%')

       

    return model, train_mape_history, test_mape_history

