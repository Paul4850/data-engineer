from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from regression_model.processing import preprocessors as pp
from regression_model.processing import features
from regression_model.config import config
from sklearn.cluster import KMeans

import logging

#KMeans(n_clusters=7, random_state=1).fit(X)
_logger = logging.getLogger(__name__)

basket_pipe = Pipeline(
    [
        ('BasketPreprocessor', pp.BasketPreprocessor()),
        ('scaler', MinMaxScaler()),
        ('KMeans', KMeans(n_clusters=7, random_state=1))
    ]
)


price_pipe = Pipeline(
    [
        ('categorical_imputer',
            pp.CategoricalImputer(variables=config.CATEGORICAL_VARS_WITH_NA)),
        ('numerical_inputer',
            pp.NumericalImputer(variables=config.NUMERICAL_VARS_WITH_NA)),
        ('temporal_variable',
            pp.TemporalVariableEstimator(
                variables=config.TEMPORAL_VARS,
                reference_variable=config.DROP_FEATURES)),
        ('rare_label_encoder',
            pp.RareLabelCategoricalEncoder(
                tol=0.01,
                variables=config.CATEGORICAL_VARS)),
        ('categorical_encoder',
            pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)),
        ('log_transformer',
            features.LogTransformer(variables=config.NUMERICALS_LOG_VARS)),
        ('drop_features',
            pp.DropUnecessaryFeatures(variables_to_drop=config.DROP_FEATURES)),
        ('scaler', MinMaxScaler()),
        ('forest', Lasso(random_state=0))
    ]
)
