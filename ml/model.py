from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from textwrap import dedent


def train_model(x_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    rf = RandomForestClassifier(
        n_estimators=100,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced',
        max_samples=0.8
    )
    rf.fit(x_train, y_train)

    return rf


def compute_model_metrics(y, preds, display=False):
    """
    Validates the trained machine learning model using precision, recall, and
    F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)

    if display:
        print(f'f-beta: {fbeta}')
        print(f'precision: {precision}')
        print(f'recall: {recall}')
        pass
    else:
        return precision, recall, fbeta


def inference(model, x):
    """
    Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.ensemble.RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    preds = model.predict(x)

    return preds


def calc_slice_performance(
    data, model, cat_features, encoder, lb
) -> str:
    '''
    Calculates the trained model performance over slices in categorical data.

    Inputs
    ------
        data: pandas.DataFrame
            Dataset containing the test data to be analyzed.
        model: sklearn.ensemble.RandomForestClassifier
            Trained Random Forest model.
        cat_features: List[str]
            List of categorial columns in the dataset.
        encoder : sklearn.preprocessing._encoders.OneHotEncoder
            Trained OneHotEncoder.
        lb : sklearn.preprocessing._label.LabelBinarizer
            Trained LabelBinarizer.
    '''

    slice_perf = 'MODEL SLICE PERFORMANCE REPORT'

    for col in cat_features:
        fixed_vals = set(data[col])
        for val in fixed_vals:
            cut = data.loc[data[col] == val, :]
            x_slice, y_slice, _, _ = process_data(
                cut,
                categorical_features=cat_features,
                label='salary',
                training=False,
                encoder=encoder,
                lb=lb
            )
            preds_slice = inference(model, x_slice)
            slice_perf += f'\n\nCategorical column:{col}\t| Slice value:{val}'
            precision, recall, fbeta = compute_model_metrics(
                y_slice, preds_slice, display=False
            )
            slice_perf += dedent(
                f'''\

                precision = {precision}
                recall = {recall}
                fbeta = {fbeta}
                '''
            )

    return slice_perf
