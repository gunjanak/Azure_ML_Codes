import torch
import argparse
import os
import joblib
from azureml.core import Run, Model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from helper import stock_dataFrame,create_sequences,normalize_data,train_model


# # Get Azure ML Run context
run = Run.get_context()
# Parse hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--symbol",type=str,default="ADBL",help="The stock symbol")
parser.add_argument("--input_size",type=int,default=1,help="input size")
parser.add_argument("--hidden_size",type=int,default=32,help="hidden size")
parser.add_argument("--output_size",type=int,default=1,help="output size")
parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
args = parser.parse_args()

#1. Get the data
stock_symbol = args.symbol

df = stock_dataFrame(stock_symbol,start_date='2020-01-01',weekly=False)
df.dropna(inplace=True)
X,y = create_sequences(df, window_size=5)
# 2. Normalize data
X_scaled, y_scaled, scaler = normalize_data(X, y)

# 3. Reshape for GRU (samples, timesteps, features)
X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, shuffle=False)

# 5. Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

 # 6. Train model
model, train_mape, test_mape = train_model(X_train, y_train, X_test, y_test, scaler, epochs=100, batch_size=32)
print("Train_MAPE",sum(train_mape)/len(train_mape))
print("Test_MAPE",sum(test_mape)/len(test_mape))
run.log("Train_MAPE",sum(train_mape)/len(train_mape))
run.log("Test_MAPE",sum(test_mape)/len(test_mape))

# 7. save model
os.makedirs("outputs",exist_ok=True)
model_path = "outputs/pytorch_gru.pth"
# torch.save(model,model_path)
torch.save(model.state_dict(),model_path)
print(f"Model saved in {model_path}")
# #register model
Model.register(
    workspace=run.experiment.workspace,
    model_name="pytorch_gru",
    model_path = model_path,
    description="adbl trained on new gru"
)
print("Model registered successfully")

# 8. Save scaler
scaler_path = "outputs/scaler.save"
joblib.dump(scaler, scaler_path)
print(f"Scaler saved at {scaler_path}")

# Register scaler
Model.register(
    workspace=run.experiment.workspace,
    model_name="scaler",
    model_path=scaler_path,
    description="Scaler used for ADBL GRU model"
)
print("Scaler registered successfully")


# 7. Plot results
plt.figure(figsize=(10, 5))
plt.plot(train_mape, label='Train MAPE')
plt.plot(test_mape, label='Test MAPE')
plt.title('MAPE Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MAPE (%)')
plt.legend()
plt.show()
# Save the plot to a file
plot_path = "LOSS.png"
plt.savefig(plot_path)
plt.close()
# Log the plot to Azure ML
run.log_image('LOSS PLOT', plot_path)

# 9. Predict on test set
model.eval()
with torch.no_grad():
    y_pred = model(X_test).squeeze().numpy()
    y_true = y_test.squeeze().numpy()

# 10. Inverse scale predictions and true values
y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test_inv = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()

# 11. Plot predictions vs true values
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label="Actual", linewidth=2)
plt.plot(y_pred_inv, label="Predicted", linestyle='--')
plt.title(f"{stock_symbol} Price Prediction (Test Set)")
plt.xlabel("Time Step")
plt.ylabel("Price")
plt.legend()
prediction_plot_path = "outputs/prediction_vs_actual.png"
plt.savefig(prediction_plot_path)
plt.close()
print(f"Prediction plot saved at {prediction_plot_path}")

# Log prediction plot to Azure ML
run.log_image("Prediction vs Actual", prediction_plot_path)



# Complete run
run.complete()