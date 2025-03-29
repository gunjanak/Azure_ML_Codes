import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from azureml.core import Run, Model
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Get Azure ML Run context
run = Run.get_context()

# Load Breast Cancer dataset
data = datasets.load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define a simple Neural Network
class BreastCancerNN(nn.Module):
    def __init__(self, input_dim):
        super(BreastCancerNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)

# Parse hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
args = parser.parse_args()

# Initialize model, loss, and optimizer
model = BreastCancerNN(input_dim=X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Initialize list to store losses for plotting
losses = []

# Train the model
for epoch in range(args.epochs):
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    # Calculate average loss for the epoch
    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    
    # Log loss to Azure ML
    run.log('Training Loss', avg_loss)
    
    # Print loss every epoch
    print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")

# Plot loss over epochs
plt.figure(figsize=(8, 5))
plt.plot(range(1, args.epochs+1), losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.grid(True)

# Save the plot to a file
plot_path = "loss_plot.png"
plt.savefig(plot_path)
plt.close()

# Log the plot to Azure ML
run.log_image('Loss Plot', plot_path)

# Evaluate model on test set
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred = (y_pred > 0.5).float()
    accuracy = (y_pred.eq(y_test_tensor).sum() / len(y_test_tensor)).item()

# Log accuracy to Azure ML
run.log("Accuracy", accuracy)

# Save model
os.makedirs("outputs", exist_ok=True)
model_path = "outputs/breast_cancer_nn.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved in {model_path} with accuracy: {accuracy:.4f}")

# Register the trained model in Azure ML
Model.register(
    workspace=run.experiment.workspace,
    model_name="breast_cancer_nn",
    model_path=model_path,
    description="Breast Cancer Classification Model using PyTorch"
)
print("Model registered successfully!")

# Complete run
run.complete()