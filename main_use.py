"""
Script to interface with the developed API, for inferencing and testing

Author: Paulo Souza
Date: Feb. 2023
"""

import json
from fastapi.testclient import TestClient
from main import app


def go() -> None:
    client = TestClient(app)

    test_obs = {
        'age': 50,
        'workclass': "Private",
        'fnlgt': 234721,
        'education': "Doctorate",
        'education_num': 16,
        'marital_status': "Separated",
        'occupation': "Exec-managerial",
        'relationship': "Not-in-family",
        'race': "Black",
        'sex': "Female",
        'capital_gain': 0,
        'capital_loss': 0,
        'hours_per_week': 50,
        'native_country': "United-States"
    }

    proc_obs = json.dumps(test_obs)
    resp = client.post("/prediction/", data=proc_obs)
    print(resp.json())

    print(f'Response status code: {resp.status_code}')


if __name__ == '__main__':
    go()
