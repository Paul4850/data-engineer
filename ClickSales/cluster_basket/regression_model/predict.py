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
_price_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction():
    """Make a prediction using a saved model pipeline.

    Args:
        input_data: Array of model prediction inputs.

    Returns:
        Predictions for each input row, as well as the model version.
    """
   #test_data = /srv/ftp/Qlik/ProductCategoryBasket.csv
    test_data = load_dataset(file_name='test.csv')


    data = pd.DataFrame(test_data)
    validated_data = validate_inputs(input_data=data)

    prediction = _price_pipe.predict(validated_data[config.FEATURES])

    output = np.exp(prediction)

    results = {'predictions': output, 'version': _version}

    _logger.info(
        f'Making predictions with model version: {_version} '
        f'Inputs: {validated_data} '
        f'Predictions: {results}')

    return results
