# Model Card


## Model Details

- Author: Paulo Souza
- Date: Feb. 2023
- Model Version: 0.0.1
- Model Type: Random Forest Classifier

## Intended Use

- Primary intended uses: Predicting the income of american population
- Primary intended users: Data scientists and analysts

## Training Data

- Datasets: census.csv -> acquired from [UCI repository](https://archive.ics.uci.edu/ml/datasets/census+income)
- Motivation: use of a comprehensive and stabilished dataset for model training
- Preprocessing: just some basic data cleaning, removing trailling whitespaces, performing One-Hot-Encoding over categorical features and Binarizing the target feature

## Evaluation Data

Same as above

## Metrics

- f-beta: 0.686
- precision: 0.756
- recall: 0.628

## Ethical Considerations

The model was trained over inbalanced data, where the categorical features were not present in the same frequency over the dataset. This may cause the model to prefer some predictions for some groups of categorical data, potentially generating some ethical concerns related to prediction fairness.