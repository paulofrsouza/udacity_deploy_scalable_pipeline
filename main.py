'''
Script for defining a RESTful API using FastAPI

Author: Paulo Souza
Date: Feb. 2023
'''


import pickle

from fastapi import FastAPI
from pydantic.main import BaseModel

import pandas as pd
from ml.data import process_data

app = FastAPI(
    title="Census API",
    description="An API for serving predictions over the US census dataset.",
    version="0.0.1")


class DataInput(BaseModel):
    '''
    Input data class for feeding into the API.
    '''
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
        '''
        DataInput example schema class
        '''
        schema_extra = {
            "example": {
                'age': 39,
                'workclass': "State-gov",
                'fnlgt': 77516,
                'education': "Bachelors",
                'education_num': 13,
                'marital_status': "Never-married",
                'occupation': "Adm-clerical",
                'relationship': "Not-in-family",
                'race': "White",
                'sex': "Male",
                'capital_gain': 2174,
                'capital_loss': 0,
                'hours_per_week': 40,
                'native_country': "United-States"
            }
        }


@app.on_event("startup")
async def startup_event():
    '''
    Function to load global objects, speeding up startup process.
    '''
    global RF_MODEL, DATA_ENCODER, LIN_BINARIZER
    try:
        with open('./model/rf_model.pkl', 'rb') as file:
            RF_MODEL = pickle.load(file)
        with open('./model/encoder.pkl', 'rb') as file:
            DATA_ENCODER = pickle.load(file)
        with open('./model/lb.pkl', 'rb') as file:
            LIN_BINARIZER = pickle.load(file)
    except Exception as err:
        print(
            'Some or all of the model prediction objects could not be loaded.'
        )
        raise err


@app.get("/")
async def welcome():
    '''
    Displays welcome message on API's root page.
    '''
    return "Welcome to the FastAPI model app"


@app.post("/inference/")
async def inference(inference: DataInput):
    '''
    API's main function. Performs inference over the passed data.
    '''
    try:
        with open('./model/rf_model.pkl', 'rb') as file:
            RF_MODEL = pickle.load(file)
        with open('./model/encoder.pkl', 'rb') as file:
            DATA_ENCODER = pickle.load(file)
        with open('./model/lb.pkl', 'rb') as file:
            LIN_BINARIZER = pickle.load(file)
    except Exception as err:
        print(
            'Some or all of the model prediction objects could not be loaded.'
        )
        raise err

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

    treated_obs, _, _, _ = process_data(
        obs,
        categorical_features=cat_features,
        training=False,
        encoder=DATA_ENCODER,
        lb=LIN_BINARIZER
    )

    pred = RF_MODEL.predict(treated_obs)
    pred = '<=50k' if pred[0] < .5 else '>50k'
    data['prediction'] = pred

    return data


if __name__ == '__main__':
    pass
