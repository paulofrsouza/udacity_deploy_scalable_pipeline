"""
Script to train a classification model for deployment.

Author: Paulo Souza
Date: Feb. 2023
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    train_model,
    compute_model_metrics,
    inference,
    calc_slice_performance
)
import pickle


def go() -> None:
    data = pd.read_csv('./data/clean_census.csv')

    train, test = train_test_split(
        data,
        test_size = 0.20,
        random_state = 42
    )

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
    x_train, y_train, encoder, lb = process_data(
        train,
        categorical_features = cat_features,
        label = "salary",
        training=True
    )

    x_test, y_test, _, _ = process_data(
        test, 
        categorical_features = cat_features, 
        label = 'salary', 
        training = False,
        encoder = encoder,
        lb = lb
    )

    model = train_model(x_train, y_train)
    
    slice_perf = calc_slice_performance(
        test,
        model,
        cat_features,
        encoder,
        lb
    )

    with open('./model/rf_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    with open('./model/encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
        
    with open('./model/lb.pkl', 'wb') as f:
        pickle.dump(lb, f)
        
    with open('./model/slice_output.txt', 'w') as f:
        f.write(slice_perf)
        
    
if __name__ == '__main__':
    go()