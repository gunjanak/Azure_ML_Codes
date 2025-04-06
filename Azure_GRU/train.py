import torch
import argparse
import os
import joblib
from azureml.core import Run, Model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from helper import stock_dataFrame,create_sequences,normalize_data,train_and_evaluate_gru


# # Get Azure ML Run context
run = Run.get_context()
# Parse hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--symbol",type=str,default="ADBL",help="The stock symbol")
parser.add_argument("--input_size",type=int,default=1,help="input size")
parser.add_argument("--hidden_size",type=int,default=32,help="hidden size")
parser.add_argument("--output_size",type=int,default=1,help="output size")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
args = parser.parse_args()

#1. Get the data
stock_symbol = args.symbol
df = stock_dataFrame(stock_symbol,start_date='2020-01-01',weekly=False)
df = df[['Close']]
df.dropna(inplace=True)

#2. Format the data 
X,y = create_sequences(df, window_size=5)

# 3. Normalize data
X_scaled, y_scaled, scaler = normalize_data(X, y)

# 4. Reshape for GRU (samples, timesteps, features)
X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# 5. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, shuffle=False)

# 6. Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")

 # 6. Train model
model, train_losses, test_losses, test_mapes = train_and_evaluate_gru(X_train, y_train, X_test, y_test, input_size=1, hidden_size=32, output_size=1, epochs=100, lr=0.001)
print("Test_MAPE",sum(test_mapes)/len(test_mapes))
run.log("Test_MAPE",sum(test_mapes)/len(test_mapes))

# 7. save model
os.makedirs("outputs",exist_ok=True)
model_path = "outputs/pytorch_nn_gru.pth"
torch.save(model.state_dict(),model_path)
print(f"Model saved in {model_path}")
# #register model
Model.register(
    workspace=run.experiment.workspace,
    model_name="pytorch_nn_gru",
    model_path = model_path,
    description="adbl trained on new gru"
)
print("Model registered successfully")


# 8. Save scaler
scaler_path = "outputs/scaler_adbl.save"
joblib.dump(scaler, scaler_path)
print(f"Scaler saved at {scaler_path}")

# Register scaler
Model.register(
    workspace=run.experiment.workspace,
    model_name="adbl_scaler",
    model_path=scaler_path,
    description="Scaler used for ADBL GRU model"
)
print("Scaler registered successfully")


# # 7. Plot results
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train LOSSES')
plt.plot(test_losses, label='Test LOSSES')
plt.title('LOSS Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('LOSS (%)')
plt.legend()
plt.show()
# Save the plot to a file
plot_path = "LOSS.png"
plt.savefig(plot_path)
plt.close()
# Log the plot to Azure ML
run.log_image('LOSS PLOT', plot_path)


# Complete run
run.complete()