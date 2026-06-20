import sys
import pandas as pd


from src.exception import CustomException
from src.utils import load_object

from src.config.paths import (
    MODEL_PATH,
    PREPROCESSOR_PATH
)

class PredictPipeline:
    def __init__(self):
            try:
                self.model=load_object(MODEL_PATH)
                self.preprocessor=load_object(PREPROCESSOR_PATH)
            except Exception as e:
                raise CustomException(e, sys)

    def predict(self, features):
        try:
            transformed_features=self.preprocessor.transform(features)
            prediction=self.model.predict(transformed_features)

            return prediction
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                gender:str,
                race_ethnicity:str,
                parental_level_of_education:str,
                lunch:str,
                test_preparation_course: str,
                reading_score:int,
                writing_score:int ):
        
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score
    
    
    def to_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race/ethnicity":[self.race_ethnicity],
                "parental level of education":[self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course":[self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score":[self.writing_score]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
        