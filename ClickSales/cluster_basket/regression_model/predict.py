import numpy as np
import pandas as pd

from regression_model.processing.data_management import load_pipeline
from regression_model.config import config
from regression_model.processing.validation import validate_inputs
from regression_model.processing.data_management import load_dataset
from regression_model import __version__ as _version
from regression_model import pipeline

import logging
import typing as t


_logger = logging.getLogger(__name__)

pipeline_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
_basket_pipe = load_pipeline(file_name=pipeline_file_name)


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
