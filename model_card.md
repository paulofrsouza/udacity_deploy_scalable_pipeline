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

The evaluation data was obtained with a 20% random sample, without replacement, of the entire dataset.

## Metrics

- f-beta: 0.686
- precision: 0.756
- recall: 0.628

## Caveats and Recommendations

The dataset used is based on a slice of census data from the american populaion. It's recommended to use the model onl for the purpose it was designed for, given its predictions may have sensitive impacts on given populational groups. Moreover, its important to use the model in the API context it was designed for.

## Ethical Considerations

The model was trained over inbalanced data, where the categorical features were not present in the same frequency over the dataset. This may cause the model to prefer some predictions for some groups of categorical data, potentially generating some ethical concerns related to prediction fairness.