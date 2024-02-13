# import pandas as pd
# from typing import List, Union, Tuple
# import spacy
# import logging
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from edu_convokit import uptake_utils
# from scipy.special import softmax
# import logging
# import re


import os
import logging
import re
import spacy
import torch
import pandas as pd
from typing import List, Union, Tuple
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizer, PreTrainedTokenizerFast
# import uptake_utils

from constants import (
    STUDENT_REASONING_HF_MODEL_NAME,
    STUDENT_REASONING_MIN_NUM_WORDS,
    STUDENT_REASONING_MAX_INPUT_LENGTH,
    FOCUSING_HF_MODEL_NAME,
    FOCUSING_MIN_NUM_WORDS,
    FOCUSING_MAX_INPUT_LENGTH,
    UPTAKE_HF_MODEL_NAME,
    UPTAKE_MIN_NUM_WORDS_SPEAKER_A,
    HIGH_UPTAKE_THRESHOLD,
    UPTAKE_MAX_INPUT_LENGTH,
    MATH_PREFIXES,
    MATH_WORDS,
    TEACHER_TALK_MOVES_HF_MODEL_NAME,
    STUDENT_TALK_MOVES_HF_MODEL_NAME
)

class Annotator:
    '''
        Annotator class derivered from the [edu-convokit](https://github.com/stanfordnlp/edu-convokit/blob/main/edu_convokit/annotation/annotator.py)
    '''

    def __init__(self) -> None:
        self.transcript = None
    
    def _initialize(self, model_shortname: str) -> (PreTrainedTokenizer | PreTrainedTokenizerFast, any):
        '''
            Returning object of the HF model
        '''
        tokenizer = AutoTokenizer.from_pretrained(model_shortname)
        model = AutoModelForSequenceClassification.from_pretrained(model_shortname)
        model.eval()
        return tokenizer, model

    def _loadData(self, filepath: str) -> pd.DataFrame:
        '''
            Load the json transcripts

            Arguments:
            - filepath:str -> path of the json file

            Output:
            @pd.DataFrame
        '''
        self.transcript = pd.read_json(
            path_or_buf= filepath,
            orient= "records",
            lines= True
        )

        return self.transcript

    def _populateAnalysisUnit(
            self,
            df: pd.DataFrame,
            analysis_unit: str,
            text_column: str,
            time_start_column: str,
            time_end_column: str,
            output_column: str
        ) -> pd.DataFrame:
        '''
            Populate output_column with number of words, sentences or timestamps
        '''

        if analysis_unit == "words":
            df[output_column] = df[text_column].str.split().str.len()
        elif analysis_unit == "sentences":

            nlp = spacy.load("en_core_web_sm")
            df[output_column] = df[text_column].apply(lambda x: len(list(nlp(x).sents)))
        elif analysis_unit == "timestamps":
            # Check type of time_start_column and time_end_column
            if df[time_start_column].dtype != "float64":
                df[time_start_column] = df[time_start_column].astype("float64")
            if df[time_end_column].dtype != "float64":
                df[time_end_column] = df[time_end_column].astype("float64")
            df[output_column] = df[time_end_column] - df[time_start_column]
        else:
            raise ValueError(f"Analysis unit {analysis_unit} not supported.")
        
        return df

    def _getClassificationPredictions(
            self,
            df: pd.DataFrame,
            text_column: str,
            output_column: str,
            model_name: str,
            min_num_words: int = 0,
            max_num_words: int = None,
            speaker_column: str = "speaker",
            speaker_value: Union[str, List[str]] = None,
    ) -> pd.DataFrame:
        """
            Get classification predictions for a dataframe

            ### Arguments:
            -    df: pandas dataframe
            -    text_column: name of column containing text to get predictions for
            -    output_column: name of column to store predictions
            -    speaker_column: name of column that contains speaker names.
            -    speaker_value: if speaker_column is not None, only get predictions for this speaker.
            -    model_name: name of model to use.
        """

        assert text_column in df.columns, f"Text column {text_column} not found in dataframe."

        print(df.columns, output_column, output_column in df.columns)

        if (output_column in df.columns) == True:
            logging.warning(f"Target column {output_column} already exists in dataframe. Skipping.")
            return df
        
        if speaker_column is not None:
            assert speaker_column in df.columns, f"Speaker column {speaker_column} not found in dataframe."

            if isinstance(speaker_value, str):
                speaker_value = [speaker_value]
        
        tokenizer, model = self._initialize(model_name)

        # Get predictions
        predictions = []
        for _, row in df.iterrows():
            # Skip if speaker doesn't match
            # if speaker_column is not None:
            #         predictions.append(None)
            #         continue
                
            text = row[text_column]

            # Skip if text is too short
            if len(text.split()) < min_num_words:
                predictions.append(None)
                continue
            
            with torch.no_grad():
                inputs = tokenizer(
                    text= text, 
                    return_tensors="pt", 
                    padding= True,
                    truncation= True,
                    max_length= max_num_words
                    )
                outputs = model(**inputs)
                logits = outputs.logits
                predictions.append(logits.argmax().item())
            

        df[output_column] = predictions
        return df

    
    def getTalkTime(
            self,
            text_column: str = "text",
            analysis_unit: str = "words", #words, sentences, timestamps
            representation: str = "frequency", # frequency
            time_start_column: str = "start",
            time_end_column: str = "end",
            output_column: str = "talktime",
    ) -> pd.DataFrame:
        '''
            Analyze talk time of speakers in a dataframe. Return original df and new dataframe with talk time analysis.

            Arguments:
                df (pd.DataFrame): dataframe to analyze
                text_column (str): name of column containing text to analyze. Only required if analysis_unit is words or sentences.
                analysis_unit (str): unit to analyze. Choose from "words", "sentences", "timestamps".
                representation (str): representation of talk time. Choose from "frequency", "proportion".
                time_start_column (str): name of column containing start time. Only required if analysis_unit is timestamps.
                time_end_column (str): name of column containing end time. Only required if analysis_unit is timestamps.
                output_column (str): name of column to store result.

            Returns:
                df (pd.DataFrame): dataframe with talk time analysis
        '''
        
        assert analysis_unit in ["words", "sentences", "timestamps"], f"Analysis unit {analysis_unit} not supported."
        assert representation in ["frequency", "proportion"], f"Representation {representation} not supported."
        assert text_column in self.transcript.columns, f"Text column {text_column} not in the provided dataframe."
        assert time_start_column in self.transcript.columns, f"Time start column {time_start_column} not found in dataframe."
        assert time_end_column in self.transcript.columns, f"Time end column {time_end_column} not found in dataframe."

        # Populating output column with num of words, sentences or timestamps
        self.transcript = self._populateAnalysisUnit(
            df = self.transcript,
            analysis_unit= analysis_unit,
            text_column= text_column,
            time_start_column= time_start_column,
            time_end_column= time_end_column,
            output_column= output_column
        )

        # Retun df  with talk time analysis
        if representation == 'proportion':
            total = self.transcript[output_column].sum()
            self.transcript[output_column] = self.transcript[output_column] / total
        
        return self.transcript
    
    def getStudentReasoning(
            self,
            text_column: str = "text",
            output_column: str = "student_reasoning",
            speaker_column: str = "speaker",
            speaker_value: Union[str, List[str]] = None
        ) -> pd.DataFrame:
        '''
            Get student reasoning prediction for a df

            Arguments:
                df (pd.DataFrame): dataframe to analyze
                text_column (str): name of column containing text to analyze
                output_column (str): name of column to store result
                speaker_column (str): name of column containing speaker names. Only required if speaker_value is not None.
                speaker_value (str or list): if speaker_column is not None, only get predictions for this speaker.
            
            Returns:
                df (pd.DataFrame): dataframe with student reasoning predictions
        '''
        # Print out note that the predictions should only be run on student reasoning as that's what the model was trained on.
        logging.warning(
            """
                Note: This model was trained on student reasoning, so it should be used on student utterances.
                For more details on the model, see https://arxiv.org/pdf/2211.11772.pdf
            """
        )

        print(self.transcript.columns)

        return self._getClassificationPredictions(
            df = self.transcript,
            text_column= text_column,
            output_column= output_column,
            model_name= STUDENT_REASONING_HF_MODEL_NAME,
            min_num_words= STUDENT_REASONING_MIN_NUM_WORDS,
            max_num_words= STUDENT_REASONING_MAX_INPUT_LENGTH,
            speaker_column= speaker_column,
            speaker_value= speaker_value
        )
    
    def getFocusingQuestions(
            self,
            text_column: str= "text",
            output_column: str= "focusing_questions",
            speaker_column: str = "speaker",
            speaker_value: Union[str, List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get focusing question predictions for a dataframe.

        Arguments:
            df (pd.DataFrame): dataframe to analyze
            text_column (str): name of column containing text to analyze
            output_column (str): name of column to store result
            speaker_column (str): name of column containing speaker names. Only required if speaker_value is not None.
            speaker_value (str or list): if speaker_column is not None, only get predictions for this speaker.

        Returns:
            df (pd.DataFrame): dataframe with focusing question predictions
        """

        logging.warning(
            """
                Note: This model was trained on teacher focusing questions, so it should be used on teacher utterances.
                For more details on the model, see https://aclanthology.org/2022.bea-1.27.pdf
            """
        )

        return self._getClassificationPredictions(
            df=self.transcript,
            text_column=text_column,
            output_column=output_column,
            model_name=FOCUSING_HF_MODEL_NAME,
            min_num_words=FOCUSING_MIN_NUM_WORDS,
            max_num_words=FOCUSING_MAX_INPUT_LENGTH,
            speaker_column=speaker_column,
            speaker_value=speaker_value
        )
    
    def _getUptakePredictions(self, model, device, instance):

        instance["attention_mask"] = [[1] * len(instance["input_ids"])]


if __name__ == "__main__":
    ann = Annotator()
    ann._loadData('data/Aaron Austen - Python (2)_forloops-ifelse/transcripts/Aaron Austen - Python (2)_forloops-ifelse.json')
    print(ann.getTalkTime())
    print(ann.getStudentReasoning())
    print(ann.getFocusingQuestions())