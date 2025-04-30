import mlflow.pyfunc
import pandas as pd

def init():
    global model
    # this path maps to AZUREML_MODEL_DIR
    model = mlflow.pyfunc.load_model(model_uri="diabetes_prod_model:1")

def run(raw_json):
    # assume the caller POSTs a JSON with orient="split" or orient="records"
    data = pd.read_json(raw_json, orient="split")  
    preds = model.predict(data)
    return preds.tolist()
