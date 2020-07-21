import pandas as pd
from sklearn import decomposition
from sklearn import datasets

import numpy as np
from sklearn.model_selection import train_test_split

from regression_model import pipeline
from regression_model.processing.data_management import (
    load_dataset, save_pipeline)
from regression_model.config import config
from regression_model import __version__ as _version

import logging


_logger = logging.getLogger(__name__)


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)

    # divide train and test
    #    X_train, X_test, y_train, y_test = train_test_split(
    #       data[config.FEATURES],
  #      data[config.TARGET],
   #     test_size=0.1,
    #    random_state=0)  # we are setting the seed here

    BASKET_FEATURES = [100010, 100015, 100016, 100017, 100018, 300062, 500811,
                       500812, 500813, 500814, 500815, 500816, 500818, 500819,
                       500821, 500822, 500823, 500825]
    print('debug')
    print(data.shape)
    data =  data.pivot_table('Quantity', ['TransactionId', 'StoreId'], 'MerchandiseId')
    data = pd.DataFrame(data.to_records())

    print('>>')
    print(data.shape)
    X_train = data.iloc[:, 2:len(data.columns)+2]
    print(X_train.shape)
    featDf = pd.DataFrame(columns=BASKET_FEATURES)
    X_train = pd.merge(X_train, featDf, how='left')
    X_train = X_train[BASKET_FEATURES]

    pipeline.basket_pipe.fit(X_train )
    #pipeline.price_pipe.fit(X_train )
    result = pipeline.basket_pipe.predict(X_train)
    print(result)

    _logger.info(f'saving model version: {_version}')
    save_pipeline(pipeline_to_persist=pipeline.basket_pipe)


if __name__ == '__main__':
    run_training()
