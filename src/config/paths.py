import os

ARTIFACTS_DIR="artifacts"

TRAIN_DATA_PATH = os.path.join(ARTIFACTS_DIR, 'train.csv')
TEST_DATA_PATH = os.path.join(ARTIFACTS_DIR, 'test.csv')
RAW_DATA_PATH = os.path.join(ARTIFACTS_DIR, 'data.csv')

MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'model.pkl')
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, 'preprocessor.pkl')
