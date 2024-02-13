#############################################
#############################################
# Author: Bakharia
# Date: Jan 9, 2024
# Name: prepare.py
# Functionality:
#############################################
#############################################
#############################################
#############################################
# Importing Libraries
# %%
import logging
from glob import glob
import os
import sys
import math
import re
import torch
import pandas as pd
import numpy as np
from pyannote.audio import Pipeline
from pyannote.core import Segment
# from pyannote.audio.pipelines.utils.hook import ProgressHook
from pydub import AudioSegment
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from constants import PUNC_SENT_END
#############################################
#############################################
#############################################
#############################################
# Start of class

# %%
class RecordPipe:
    '''
    RecordPipe Class
    '''

    # PUNC_SENT_END = ['.', '?', '!']

    def __init__(self, recording_dir: str) -> None:
        """
        Initialize the RecordPipe class.

        Parameters:
        - recording_dir (str): Directory containing audio recordings.
        """
        load_dotenv()
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.recording_dir = recording_dir
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.environ['HUGGING_FACE_KEY']
        )
        self.pipeline.to(torch.device("cuda"))

    #############################################
    #############################################
    # Modified from pyannote_whisper/utils.py
    # Link: https://github.com/yinruiqing/pyannote-whisper

    def _getTextWithTimeStamp(self, transcribe_res) -> list:
        """
        Extract timestamped text from the transcribed result.

        Parameters:
        - transcribe_res: Transcription result from the Whisper API.

        Returns:
        - list: List of tuples containing timestamped text.
        """
        timestamp_texts = []

        for item in transcribe_res.segments:
            start = item['start']
            end = item['end']
            text = item['text']
            timestamp_texts.append(
                (
                    Segment(start, end),
                    text
                )
            )

        return timestamp_texts

    def _addSpeakerInfoToText(self, timestamp_texts: list, ann) -> list:
        """
        Add speaker information to the timestamped text.

        Parameters:
        - timestamp_texts (list): List of timestamped text.
        - ann: Diarization result from pyannote.

        Returns:
        - list: List of tuples containing timestamped text with speaker information.
        """
        spk_text = []

        for seg, txt in timestamp_texts:
            spk = ann.crop(seg).argmax()
            spk_text.append((seg, spk, txt))

        return spk_text

    def _mergeCache(self, text_cache: list):
        """
        Merge consecutive text segments from the same speaker.

        Parameters:
        - text_cache (list): List of text segments.

        Returns:
        - tuple: Merged text segment.
        """
        sentence = ''.join([item[-1] for item in text_cache])
        spk = text_cache[0][1]
        start = text_cache[0][0].start
        end = text_cache[-1][0].end
        return Segment(start, end), spk, sentence

    def _mergeSentence(self, spk_text: list):
        """
        Merge consecutive text segments based on speaker.

        Parameters:
        - spk_text (list): List of text segments with speaker information.

        Returns:
        - list: Merged text segments.
        """
        merged_spk_text = []
        pre_spk = None
        text_cache = []
        for seg, spk, text in spk_text:
            if spk != pre_spk and pre_spk is not None and len(text_cache) > 0:
                merged_spk_text.append(self._mergeCache(text_cache))
                text_cache = [(seg, spk, text)]
                pre_spk = spk

            elif text and len(text) > 0 and text[-1] in PUNC_SENT_END:
                text_cache.append((seg, spk, text))
                merged_spk_text.append(self._mergeCache(text_cache))
                text_cache = []
                pre_spk = spk
            else:
                text_cache.append((seg, spk, text))
                pre_spk = spk
        if len(text_cache) > 0:
            merged_spk_text.append(self._mergeCache(text_cache))
        return merged_spk_text

    def _diarizeText(self, transcribe_res, diarization_result):
        """
        Process transcribed and diarized results to obtain merged text segments.

        Parameters:
        - transcribe_res: Transcription result from the Whisper API.
        - diarization_result: Diarization result from pyannote.

        Returns:
        - list: Merged text segments.
        """
        timestamp_texts = self._getTextWithTimeStamp(transcribe_res)
        spk_text = self._addSpeakerInfoToText(
            timestamp_texts= timestamp_texts,
            ann= diarization_result
        )
        res_processed = self._mergeSentence(spk_text)

        return res_processed

    def _writeToText(self, spk_sent, file_path: str):
        """
        Write merged text segments to a text file.

        Parameters:
        - spk_sent (list): Merged text segments.
        - file_path (str): Path to the output text file.
        """
        with open(file_path, 'w', encoding='utf-8') as fp:
            for seg, spk, sentence in spk_sent:
                line = f"{seg.start:.2f} {seg.end:.2f} {spk} {sentence} \n"
                fp.write(line)
    #############################################
    #############################################
    #############################################
    #############################################
    def _transcribe(self, file):
        """
        Transcribe audio content using OpenAI's Whisper API.

        Parameters:
        - file: Audio file to transcribe.

        Returns:
        - str: The transcribed text.
        """
        transcript = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=file,
            response_format= "verbose_json"
        )

        return transcript

    def _process_segment(self, i = -1, file_ = None, file_path = None):
        """
        Process a segment of the audio file.

        Parameters:
        - i (int): Segment index.
        - file: Audio file to process.
        - file_path (str): Path to the audio file.
        """
        five_minutes = 5 * 60 * 1000  # milliseconds

        os.makedirs(f"./data/{file_path}/recordings", exist_ok=True)
        os.makedirs(f"./data/{file_path}/transcripts", exist_ok=True)

        if i == -1:

            file_cut = file_[i * five_minutes: (i + 1) * five_minutes]
            audio_data = np.array(file_cut.get_array_of_samples())
            enhanced_audio_data = audio_data * 1.5  # Adjust the scaling factor as needed

            enhanced_audio = AudioSegment(
                enhanced_audio_data.tobytes(),
                frame_rate=file_cut.frame_rate,
                sample_width=file_cut.sample_width,
                channels=file_cut.channels
            )

            enhanced_audio.export(
                out_f=f"./data/{file_path}/recordings/{file_path}_{i}.wav",
                format="wav"
            )

            diarization_result = self.pipeline(
                    f"./data/{file_path}/recordings/{file_path}_{i}.wav",
                    min_speakers=2,
                    max_speakers=5,
                )

            file_cut = open(f"./data/{file_path}/recordings/{file_path}_{i}.wav", "rb")
            whisper_result = self._transcribe(file=file_cut)

            final_result = self._diarizeText(
                transcribe_res=whisper_result,
                diarization_result=diarization_result
            )

            for seg, spk, sent in final_result:
                line = f'{seg.start:.2f} {seg.end:.2f} {spk} {sent}'
                print(line)

            self._writeToText(final_result, file_path=f"./data/{file_path}/transcripts/{file_path}_{i}.txt")
        
        else:
            pass

          

    def _create_batch(self, file_: AudioSegment, file_path: str) -> None:
        """
        Process the entire audio file.

        Parameters:
        - file: Audio file to process.
        - file_path (str): Path to the audio file.
        """
        for i in range(math.ceil(len(file_) / (5 * 60 * 1e3))):
            self._process_segment(i, file_, file_path)

    def _traverse(self):
        """
        Traverse the directory, process each audio file, and generate transcripts.
        """
        for f in glob(f'{self.recording_dir}/*.mp4'):

            file_ = AudioSegment.from_file(
                file= f,
                format= 'mp4'
            )
            self._create_batch(
                file_= file_,
                file_path= f.split('/')[-1].split('.')[0]
            )

    def generate(self, traverse: bool = True):
        """
        Generate transcripts for audio recordings.

        Parameters:
        - traverse (bool): If True, traverse the directory and process audio files.
        """
        if traverse:
            self._traverse()

        paths = glob("data/*/transcripts")

        paths.sort()

        for p in paths:

            txt_files = glob(f"{p}/*.txt")

            txt_files.sort()

            dictionary = {
                'start': [],
                'end': [],
                'speaker': [],
                'text': []
            }

            five_minutes= 5 * 60
            iter_ = 0

            for file in tqdm(txt_files, desc = "Combining Transcripts", unit = "Split Text"):

                with open(f'{file}', mode= 'r', encoding='utf-8') as f:
                    for line in f:
                        row_split = line.split()
                        row_split = [x for x in row_split if x != '']
                        dictionary['start'].append(float(row_split[0]) + five_minutes * iter_)
                        dictionary['end'].append(float(row_split[1]) + five_minutes * iter_)
                        dictionary['speaker'].append(row_split[2])
                        dictionary['text'].append(re.sub(r'\n', '', ' '.join(row_split[3:])))

                # Update start and end based on the last values in the current file
                iter_ = iter_ + 1

                self.logger.info(f"Combing transcript: {file}")

            print(f"{p}/{p.split('/')[-2]}.json")
            pd.DataFrame(dictionary).to_json(f"{p}/{p.split('/')[-2]}.json", orient="records", lines= True)
#############################################
#############################################
# %%
if __name__ == "__main__":
    temp = RecordPipe(recording_dir='./recordings')
    # temp._traverse()
    temp.generate(traverse= False)