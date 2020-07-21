import numpy as np
import pandas as pd

from regression_model.processing.data_management import load_pipeline
from regression_model.config import config
from regression_model.processing.validation import validate_inputs
from regression_model.processing.data_management import load_dataset
from regression_model import __version__ as _version
from regression_model import pipeline
from pandas.io.json import json_normalize

import logging
import typing as t

def convert_input(jsonData) -> dict:
    #res = pd.read_json(jsonData, orient='records')
    res = pd.DataFrame(jsonData)
    print(res.shape)
    return res


_logger = logging.getLogger(__name__)

pipeline_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
_basket_pipe = load_pipeline(file_name=pipeline_file_name)

BASKET_FEATURES = [100010, 100015, 100016, 100017, 100018,
                   300057, 300058, 300060, 300061, 300062,
                   300064, 300065, 300570, 300640, 500811,
                   500812, 500813, 500814, 500815, 500816,
                   500818, 500819, 500821, 500822, 500823,
                   500825, 500827]

def make_predict2(input_data:t.Union[pd.DataFrame, dict],
                 ) -> dict:

    x_raw = pd.DataFrame(input_data)

    xx  = x_raw.pivot_table('Quantity', ['TransactionId', 'StoreId'], 'MerchandiseId')
    xx_index = xx.index

    all_features = list(set(BASKET_FEATURES + list(xx.columns)))

    featDf = pd.DataFrame(columns=all_features)

    xx = pd.merge(xx, featDf, how='left')
    xx = xx[BASKET_FEATURES]

    xx = xx.fillna(0)

    prediction = _basket_pipe.predict(xx)

    rr = _basket_pipe.predict(xx)
    predicts = pd.DataFrame(data=rr, index=xx_index, columns=['ClusterId'])
    predicts = predicts.reset_index()
    predicts = predicts.to_json(orient='records')
    results = {'predictions': predicts, 'version': _version}

    return predicts

#input example
#{""StoreId"":150,""TransactionId"":2,""MerchandiseId"":300062,""Quantity"":2},
# {""StoreId"":150,""TransactionId"":2,""MerchandiseId"":500811,""Quantity"":1}

def make_predict(input_data:t.Union[pd.DataFrame, dict],
                 ) -> dict:
    data = pd.DataFrame(input_data)
    print(input_data.shape)
    prediction = _basket_pipe.predict(data)
    results = {'predictions': prediction, 'version': _version}
    return results

def make_prediction():
    """Make a prediction using a saved model pipeline.

    Args:
        input_data: Array of model prediction inputs.

    Returns:
        Predictions for each input row, as well as the model version.
    """
   #test_data = /srv/ftp/Qlik/ProductCategoryBasket.csv
    #test_data = load_dataset(file_name='test.csv')
    data = load_dataset(file_name='ProductCategoryBasket.csv')
    data =  data.pivot_table('Quantity', ['TransactionId', 'StoreId'], 'MerchandiseId')
    data = pd.DataFrame(data.to_records())

#data = load_dataset(file_name=config.TRAINING_DATA_FILE)
    baskets = data.copy()
    #baskets =  baskets.dropna(axis = 1, thresh= len(baskets.index) * 0.01)
    baskets = baskets.iloc[:, 2:len(baskets.columns)+2]
    baskets = baskets.dropna(axis = 0, thresh= 1)
    baskets = baskets.fillna(0)

    #data = pd.DataFrame(test_data)
    #validated_data = validate_inputs(input_data=data)

    prediction = _basket_pipe.predict(baskets)

    output = prediction
    #np.exp(prediction)

    results = {'predictions': output, 'version': _version}

    _logger.info(
        f'Making predictions with model version: {_version} '
        f'Inputs: {baskets} '
        f'Predictions: {results}')

    return results
