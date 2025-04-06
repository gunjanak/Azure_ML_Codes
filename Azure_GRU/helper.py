import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from azureml.core import Run




#function to load data from api
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


    
class GRUModel(nn.Module):
    
    def __init__(self,input_size,hidden_size,output_size):
        super(GRUModel,self).__init__()
        self.gru = nn.GRU(input_size,hidden_size,batch_first=True)
        self.fc = nn.Linear(hidden_size,output_size)
        
    def forward(self,x):
        _,h_n = self.gru(x)
        return self.fc(h_n.squeeze(0))

    

def train_and_evaluate_gru(X_train, y_train, X_test, y_test, input_size=1, hidden_size=32, output_size=1, epochs=100, lr=0.001):
    model = GRUModel(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []
    test_mapes = []

    for epoch in range(epochs):
        print(f"Epoch no: {epoch+1}")
        model.train()
        train_preds = model(X_train)
        loss = criterion(train_preds.squeeze(), y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_preds = model(X_test)
            test_loss = criterion(test_preds.squeeze(), y_test)
            test_mape = mean_absolute_percentage_error(y_test.numpy(), test_preds.squeeze().numpy())
            
            
        train_losses.append(loss.item())
        test_losses.append(test_loss.item())
        test_mapes.append(test_mape)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:03d}: Train Loss={loss.item():.4f}, Test Loss={test_loss.item():.4f}, Test MAPE={test_mape:.2f}%")

    return model, train_losses, test_losses, test_mapes