
import json
import mlflow.pyfunc
import numpy as np
from azureml.core import Model

def init():
    global model
    model_dir = Model.get_model_path("best_cancer_model")  # Gets latest version automatically
    model = mlflow.pyfunc.load_model(model_dir)

def run(data):
    try:
        input_data = json.loads(data)["data"]
        predictions = model.predict(np.array(input_data))
        return {"predictions": predictions.tolist()}
    except Exception as e:
        return {"error": str(e)}
