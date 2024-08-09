import pathlib 
import os
import prediction_model

PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT,"datasets")

TRAIN_FILE = 'train.csv'
TEST_FILE = 'train.csv'

MODEL_NAME = 'Survive_classification.pkl'

SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT,'trained_models')

TARGET = 'Survived'

FEATURES = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

CAT_FEATURES = ['Pclass', 'Sex', 'Embarked']

NUM_FEATURES  = ['Age', 'SibSp', 'Parch', 'Fare']

FEATURE_TO_MODIFY = ['Parch']

FEATURE_TO_ADD  = 'SibSp'

FEATURE_DATA_TYPE_CHANGE = ['Pclass']

NUM_VARS_WITH_NA = ['Age']

CAT_VAR_WITH_NA = ['Embarked']

DROP_FEATURES = ['Name','PassengerId','Ticket','Cabin']

FEATURES_TO_ENCODE = ['Pclass', 'Sex', 'Embarked']