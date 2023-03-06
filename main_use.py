"""
Script to interface with the developed API, for inferencing and testing

Author: Paulo Souza
Date: Mar. 2023
"""

import json
import requests


def req_api() -> None:
    '''
    Sends post requests to deployed API in order to get back predictions.
    '''

    url = 'https://udacity-census-fastapi-app.onrender.com/inference'

    test_obs = {
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

    proc_obs = json.dumps(test_obs)
    resp = requests.post(url, data=proc_obs)

    print(f'Response status code: {resp.status_code}')
    print(resp.json())


if __name__ == '__main__':
    req_api()
