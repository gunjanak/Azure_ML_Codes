import json
import torch
import torch.nn as nn
import numpy as np
import os
import joblib
from azureml.core.model import Model


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n.squeeze(0))


# Global model and scaler objects
model = None
scaler = None


def init():
    global model, scaler

    # Load GRU model
    model_path = Model.get_model_path("pytorch_gru")
    model = GRUModel(input_size=1, hidden_size=32, output_size=1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    # Load scaler
    scaler_path = Model.get_model_path("scaler")
    scaler = joblib.load(scaler_path)




def run(raw_data):
    try:
        print("Received raw data:", raw_data)  # Debugging

        data = json.loads(raw_data)
        inputs = data["data"]  # expects [[val1, val2, val3, val4, val5], ...]

        # Ensure all sequences have exactly 5 timesteps
        for seq in inputs:
            if len(seq) != 5:
                return {"error": "Each input sequence must have exactly 5 timesteps."}

        # Convert to numpy array and reshape for scaling
        inputs_np = np.array(inputs, dtype=np.float32).reshape(-1, 1)
        print("Reshaped input array for scaling:", inputs_np.shape)  # Debugging

        # Apply scaling
        inputs_scaled = scaler.transform(inputs_np)
        print("Scaled inputs:", inputs_scaled.shape)  # Debugging

        # Reshape back to (batch_size, 5, 1)
        batch_size = len(inputs)
        x = torch.tensor(inputs_scaled.reshape(batch_size, 5, 1), dtype=torch.float32)
        print("Final tensor shape for model:", x.shape)  # Debugging

        with torch.no_grad():
            predictions = model(x)

        # Convert predictions back to numpy
        preds_np = predictions.numpy().reshape(-1, 1)
        print("Raw model predictions:", preds_np)  # Debugging

        # Apply inverse transformation
        preds_inversed = scaler.inverse_transform(preds_np)
        print("Inverse-scaled predictions:", preds_inversed)  # Debugging

        return preds_inversed.flatten().tolist()

    except Exception as e:
        print("Error occurred:", str(e))  # Debugging
        return {"error": str(e)}

