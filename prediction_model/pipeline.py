from sklearn.pipeline import Pipeline
from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
import prediction_model.processing.preprocessing as pp 
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

classification_pipeline = Pipeline(
    [
        ('DomainProcessing',pp.DomainProcessing(variable_to_modify = config.FEATURE_TO_MODIFY,
        variable_to_add = config.FEATURE_TO_ADD)),
        ('MeanImputation', pp.MeanImputer(variables=config.NUM_VARS_WITH_NA)),
        ('ModeImputation',pp.ModeImputer(variables=config.CAT_VAR_WITH_NA)),
        ('DropFeatures', pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
        ('LabelEncoder',pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
        
        ('MinMaxScale', StandardScaler()),
        ('LogisticClassifier',LogisticRegression(random_state=0))
    ]
)


print('DOne!!!!!!!!!!!!!!!!!!!!!!!!')