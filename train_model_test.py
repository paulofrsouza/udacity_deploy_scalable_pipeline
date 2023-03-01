'''
Module to test the 'train_model.py' script.

Author: Paulo Souza
Data: Feb. 2023
'''

import os
import logging
import pytest

from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

logging.basicConfig(
    filename='./model/train_model.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s',
    force=True)


@pytest.fixture(scope='module', name='get_data')
def get_data_():
    '''
    Fixture for fetching the inital data from a csv file.
    '''
    try:
        df_raw = pd.read_csv("./data/clean_census.csv")
        logging.info("SUCCESS - get_data: The file exists")
    except FileNotFoundError as err:
        logging.error(" - get_data: The file wasn't found")
        raise err

    return df_raw


@pytest.fixture(scope='module', name='split_data')
def split_data_(get_data):
    '''
    Fixture for splitting the dataset in train and test samples.
    '''
    df_raw = get_data.copy()
    try:
        train, test = train_test_split(
            df_raw,
            test_size = 0.20,
            random_state = 42
        )
        logging.info('SUCCESS - split_data: train and test datasets were created')
    except Exception as err:
        logging.error(' - split_data: the data coud not be splitted')
        raise err

    return train, test


@pytest.fixture(scope='module', name='process_data')
def process_data_(split_data):
    '''
    Fixture for performing final feature engineering over split data.
    '''
    train, test = split_data
    
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
        x_train, y_train, encoder, lb = process_data(
            train,
            categorical_features = cat_features,
            label = "salary",
            training=True
        )
        logging.info('SUCCESS - process_data: train data was treated')
    except Exception as err:
        logging.error(' - process_data: train data could not be treated')
        raise err
    
    try:
        x_test, y_test, _, _ = process_data(
            test, 
            categorical_features = cat_features, 
            label = 'salary', 
            training = False,
            encoder = encoder,
            lb = lb
        )
        logging.info('SUCCESS - process_data: test data was treated')
    except Exception as err:
        logging.error(' - process_data: test data could not be treated')
        raise err
    
    return x_train, y_train, x_test, y_test, encoder, lb


@pytest.fixture(scope='module', name='train_model')
def train_model_(process_data):
    '''
    Fixture for training the Random Forest model.
    '''
    x_train, y_train, _, _, _, _ = process_data
    
    try:
        model = train_model(x_train, y_train)
        logging.info('SUCCESS - train_model: the Random Forest was trained')
    except Exception as err:
        logging.error(' - train_model: the Random Forest coudl not be trained')
        raise err
    
    return model


@pytest.fixture(scope='module', name='get_preds')
def get_preds_(process_data, train_model):
    '''
    Fixture to get model predictions.
    '''
    _, _, x_test, _, _, _ = process_data
    model = train_model
    
    try:
        preds = inference(model, x_test)
        logging.info('SUCCESS - get_preds: the predictions were generated')
    except Exception as err:
        logging.error(' - get_preds: the predictions could not be generated')
        raise err

    return preds


@pytest.fixture(scope='module', name='get_metrics')
def get_metrics_(process_data, get_preds):
    '''
    Fixture to calculate model performance metrics.
    '''
    preds = get_preds.copy()
    _, _, _, y_test, _, _ = process_data
    
    try:
        precision, recall, fbeta = compute_model_metrics(y_test, preds)
        logging.info('SUCCESS - get_metrics: model performance was calculated')
    except Exception as err:
        logging.error(' - get_metrics: model performance coud not be calculated')
        raise err
        
    return precision, recall, fbeta


def test_train_model(train_model):
    '''
    Test for trained Random Forest.
    '''
    model = train_model
    rf = RandomForestClassifier()
    
    try:
        assert type(model) == type(rf)
    except AssertionError as err:
        logging.error(' - test_train_model: the model was not correctly trained')
        raise err
        
    logging.info('SUCCESS - test_train_model: the model was correctly trained')


def test_inferece(get_preds):
    '''
    Test for model inference.
    '''
    preds = get_preds.copy()
    
    try:
        assert len(preds) > 0
    except AssertionError as err:
        logging.error(' - test_inference: the model did not generate predictions')
        raise err
        
    logging.info('SUCCESS - test_inferece: the model generated predictions')
    

def test_compute_model_metrics(get_metrics):
    '''
    Test for model performance metrics calculation.
    '''
    precision, recall, fbeta = get_metrics
    
    try:
        assert precision >= 0.0 and precision <= 1.0
        assert recall >= 0.0 and recall <= 1.0
        assert fbeta >= 0.0 and fbeta <= 1.0
    except AssertionError as err:
        logging.error(' - test_model_metrics: performance metrics were not calculated')
        raise err
        
    logging.info('SUCCESS - test_model_metrics: performance metrics calculated')


def main():
    '''
    main pipeline function - used to run the file without pytest
    '''
    df_raw = get_data_()
    train, test = split_data_(df_raw)
    x_train, y_train, x_test, y_test, encoder, lb = process_data(train, test)
    model = train_model_(x_train, y_train)
    preds = get_preds_(x_test, model)
    precision, recall, fbeta = get_metrics_(preds, y_test)
    
    test_train_model(model)
    test_inference(preds)
    test_compute_model_metrics(precision, recall, fbeta)
    
    
if __name__ == '__main__':
    main()

