# Library imports
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import uvicorn
import shap
import json
from sklearn.pipeline import Pipeline

class Item(BaseModel):
    EXT_SOURCE_2: float = 0.5202729458
    EXT_SOURCE_3:float = 0.537070
    CODE_GENDER:float = 1
    PAYMENT_RATE:float = 0.0482779168
    DAYS_EMPLOYED:float = 4.52
    INSTAL_DPD_MEAN:float = 0.5333333333
    PREV_CNT_PAYMENT_MEAN:float = 12.0
    NAME_EDUCATION_TYPE_Highereducation:float = 1
    DAYS_BIRTH:float = 55.8
    AMT_ANNUITY:float = 36459.0


# Loading the model and data
model = pickle.load(open('api/best_model.pkl', 'rb'))
pipeline_preprocess = pickle.load(open('api/pipeline_preprocess.pkl', 'rb'))
explainer = shap.TreeExplainer(model)

# Create a FastAPI instance
app = FastAPI()

# Functions
@app.get('/')
def home():

    return 'Welcome to API Pret a depenser'

class ClientData(BaseModel):
    data: str

@app.post('/prediction_manual')
async def prediction_manual(item: Item):
    # Check if the trx label is in the url
    df = pd.DataFrame(item.dict(), index=[0])
    new_df = pipeline_preprocess.transform(df)
    print(df)
    try:
        results = model.predict_proba(new_df)
        return {'predic_proba': results[0][1].round(2)}
    except:
        return 'Error: No id field provided. Please specify a label.'
    


@app.post("/predict")
def predict_credit(client_data: ClientData):
    
    client_null = pd.read_json(client_data.data)
    client_df = pipeline_preprocess.transform(client_null)
    prediction = model.predict_proba(client_df)[0][1].round(2)
    return {"proba": prediction}


@app.post('/shaplocal/')
def shap_values_local(client_data: ClientData):
   
    client_null = pd.read_json(client_data.data)
    client_df = pipeline_preprocess.transform(client_null)
    shap_val = explainer(client_df)[0][:, 1]

    return {'shap_values': shap_val.values.tolist(),
            'base_value': shap_val.base_values}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
# uvicorn api:app --reload
