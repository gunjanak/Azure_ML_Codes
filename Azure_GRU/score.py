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
    model_path = Model.get_model_path("pytorch_nn_gru")
    model = GRUModel(input_size=1, hidden_size=32, output_size=1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    # Load scaler
    scaler_path = Model.get_model_path("adbl_scaler")
    scaler = joblib.load(scaler_path)


def run(raw_data):
    try:
        data = json.loads(raw_data)
        inputs = data["data"]  # expects [[val1, val2, val3, val4, val5], ...]

        # Convert to tensor with shape (batch_size, 5, 1)
        x = torch.tensor(inputs, dtype=torch.float32).unsqueeze(-1)

        with torch.no_grad():
            predictions = model(x)

        preds_np = predictions.numpy().reshape(-1, 1)

        # Inverse transform using the scaler if y was scaled
        preds_inversed = scaler.inverse_transform(preds_np)

        return preds_inversed.flatten().tolist()

    except Exception as e:
        return {"error": str(e)}
