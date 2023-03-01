'''
Script for defining a RESTful API using FastAPI

Author: Paulo Souza
Date: Feb. 2023
'''


from typing import Union, Optional
import os 
import pickle

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import pandas as pd
from ml.data import process_data

app = FastAPI(
    title = "Census API",
    description = "An API for serving predictions over the US census dataset.",
    version = "0.0.1")

class data_input(BaseModel):
    age: int
    workclass: str 
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
                'age':50,
                'workclass':"Private", 
                'fnlgt':234721,
                'education':"Doctorate",
                'education_num':16,
                'marital_status':"Separated",
                'occupation':"Exec-managerial",
                'relationship':"Not-in-family",
                'race':"Black",
                'sex':"Female",
                'capital_gain':0,
                'capital_loss':0,
                'hours_per_week':50,
                'native_country':"United-States"
            }
        }

        
@app.on_event("startup")
async def startup_event(): 
    global model, encoder, lb
    try:
        with open('./model/rf_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('./model/encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        with open('./model/lb.pkl', 'rb') as f:
            lb = pickle.load(f)
    except Exception as err:
        print('Some or all of the model prediction objects could not be loaded.')
        raise err

    
@app.get("/")
async def welcome():
    return "Welcome to the FastAPI model app"


@app.post("/inference/")
async def ingest_data(inference: data_input):
    data = {  
        'age': inference.age,
        'workclass': inference.workclass, 
        'fnlgt': inference.fnlgt,
        'education': inference.education,
        'education-num': inference.education_num,
        'marital-status': inference.marital_status,
        'occupation': inference.occupation,
        'relationship': inference.relationship,
        'race': inference.race,
        'sex': inference.sex,
        'capital-gain': inference.capital_gain,
        'capital-loss': inference.capital_loss,
        'hours-per-week': inference.hours_per_week,
        'native-country': inference.native_country,
    }

    obs = pd.DataFrame(data, index=[0])

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    try:
        with open('./model/rf_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('./model/encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        with open('./model/lb.pkl', 'rb') as f:
            lb = pickle.load(f)
    except Exception as err:
        print('Some or all of the model prediction objects could not be loaded.')
        raise err
        
    treated_obs,_,_,_ = process_data(
        obs, 
        categorical_features=cat_features, 
        training=False, 
        encoder=encoder, 
        lb=lb
    )

    pred = model.predict(treated_obs)
    pred = '<=50k' if pred[0] < .5 else '>50k'
    data['prediction'] = pred

    return data


if __name__ == '__main__':
    pass