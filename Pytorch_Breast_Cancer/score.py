import json
import torch
import torch.nn as nn
import numpy as np
from azureml.core.model import Model

# Define the model class (must match training script)
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

# Load model
def init():
    global model
    
    # Retrieve the model path from Azure ML
    model_path = Model.get_model_path("breast_cancer_nn")
    
    # Initialize model
    model = BreastCancerNN(input_dim=30)  # 30 features in the dataset
    model.load_state_dict(torch.load(model_path))
    model.eval()

# Run inference
def run(raw_data):
    try:
        # Parse input JSON data
        data = json.loads(raw_data)
        input_data = np.array(data["data"])  # Expecting key "data"
        
        # Convert to tensor
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        
        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)
            predictions = (output > 0.5).int().numpy().tolist()
        
        return json.dumps({"predictions": predictions})
    except Exception as e:
        return json.dumps({"error": str(e)})
