'''
Script for testing the API app defined on ./main.py

Author: Paulo Souza
Date: Mar. 2023
'''

import json
import logging
from fastapi.testclient import TestClient

from main import app

logging.basicConfig(
    filename='./main_test_log.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s',
    force=True
)

client = TestClient(app)


def test_welcome():
    """
    GET test - checks if the welcome message is correctly displayed.
    """
    resp = client.get("/")

    try:
        assert resp.status_code == 200
        assert resp.json() == "Welcome to the FastAPI model app"
    except AssertionError as err:
        logging.error(
            ' - test_welcome: the welcome message was not correctly returned.'
        )
        raise err

    logging.info(
        'SUCCESS - test_welcome: the welcome message was correctly returned.'
    )


def test_inference_example():
    """
    POST test - checks if the example data point is correctly predicted.
    """

    obs = {
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
    obs = json.dumps(obs)
    resp = client.post("/inference/", data=obs)

    try:
        assert resp.status_code == 200
        assert resp.json()["education-num"] == 13
        assert resp.json()["occupation"] == 'Adm-clerical'
    except AssertionError as err:
        logging.error(
            ' - test_inference_example: the data point was not correctly \
            retrieved'
        )
        raise err
    logging.info(
        'SUCCESS - test_inference_example: the data point was correctly \
        retrieved.'
    )

    try:
        assert resp.json()["prediction"] == '<=50k'
    except AssertionError as err:
        logging.info(
            ' - test_inference_example: the returned prediction was incorrect.'
        )
        raise err
    logging.info(
        'SUCCESS - test_inference_example: the returned prediction was correct'
    )


def test_inference_sample():
    """
    POST test - checks if a random data point is correctly predicted.
    """

    # observation on position 12116 in clean_census.csv dataset
    obs = {
        'age': 43,
        'workclass': "Private",
        'fnlgt': 484861,
        'education': "Some-college",
        'education_num': 10,
        'marital_status': "Married-civ-spouse",
        'occupation': "Craft-repair",
        'relationship': "Husband",
        'race': "White",
        'sex': "Male",
        'capital_gain': 4064,
        'capital_loss': 0,
        'hours_per_week': 38,
        'native_country': "United-States"
    }
    obs = json.dumps(obs)
    resp = client.post("/inference/", data=obs)

    try:
        assert resp.status_code == 200
        assert resp.json()["education-num"] == 10
        assert resp.json()["occupation"] == 'Craft-repair'
    except AssertionError as err:
        logging.error(
            ' - test_inference_sample: the data point was not correctly \
            retrieved'
        )
        raise err
    logging.info(
        'SUCCESS - test_inference_sample: the data point was correctly \
        retrieved.'
    )

    try:
        assert resp.json()["prediction"] == '<=50k'
    except AssertionError as err:
        logging.info(
            ' - test_inference_sample: the returned prediction was incorrect.'
        )
        raise err
    logging.info(
        'SUCCESS - test_inference_sample: the returned prediction was correct'
    )


if __name__ == '__main__':
    test_welcome()
    test_inference_example()
    test_inference_sample()
