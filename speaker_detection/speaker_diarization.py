from audio import preprocess_wav
from audio_io import read_wav, read_wav_as_blocks, read_wav_from_bytes, convert_to_wav
from params import *
from speech_encoder import VoiceEncoder
from demo_utils import interactive_diarization
import numpy as np


class SpeakerDiarization(object):
    def __init__(self):

        self.encoder = VoiceEncoder("cpu")
        print("Running the continuous embedding on cpu, this might take a while...")

    @staticmethod
    def convert_wav(audio_file: str, output_file_name: str = "data/youtube_test.wav", convert:bool=False, **kwargs):
        if output_file_name is None:
            output_file_name = audio_file

        audio_data = read_wav(audio_file, **kwargs)
        audio_data = preprocess_wav(audio_data)

        if convert:
            wav_file = convert_to_wav(audio_data, output_file_name)
            return wav_file
        else:
            return audio_data

    # Reference speaker segments
    def reference_speakers_embeddings(self, reference_audio_file):
        wav = self.convert_wav(audio_file=reference_audio_file)

        segments = [[0, 5.5], [6.5, 12], [17, 25]]
        speaker_names = ["Kyle Gass", "Sean Evans", "Jack Black"]
        speaker_wavs = [wav[int(s[0] * sampling_rate):int(s[1]) * sampling_rate] for s in segments]

        speaker_embeds = [self.encoder.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]
        return speaker_embeds, speaker_names

    def simple_diarize(self, audio_file, reference_audio_file=None, visualization: bool=True):

        audio_data = self.convert_wav(audio_file=audio_file, source_sr=None)
        speaker_embeds, speaker_names = self.reference_speakers_embeddings(reference_audio_file=reference_audio_file)
        # Compare speaker embeds to the continuous embedding of the interview
        # Derive a continuous embedding of the audio. We put a rate of 16, meaning that an
        # embedding is generated every 0.0625 seconds. It is good to have a higher rate for speaker
        # diarization, but it is not so useful for when you only need a summary embedding of the
        # entire utterance.

        _, cont_embeds, wav_splits = self.encoder.embed_utterance(audio_data, return_partials=True, rate=16)

        # Get the continuous similarity for every speaker.
        # Dot product should give the similarity score between audio embedding and speaker embedding
        similarity_dict = {name: cont_embeds @ speaker_embed for name, speaker_embed in
                           zip(speaker_names, speaker_embeds)}

        # Run the interactive demo
        if visualization:
            interactive_diarization(similarity_dict, audio_data, wav_splits, show_time=True)

        return similarity_dict

    def streaming_diarize(self, audio_file, reference_audio_file=None):
        wav_file = self.convert_wav(audio_file, convert=True)
        wavs = read_wav_as_blocks(audio_file=wav_file)

        speaker_embeds, speaker_names = self.reference_speakers_embeddings(reference_audio_file=reference_audio_file)
        for wav in wavs:
            _, cont_embeds, wav_splits = self.encoder.embed_utterance(wav, return_partials=True, rate=4)

            similarity_dict = {name: cont_embeds @ speaker_embed for name, speaker_embed in
                               zip(speaker_names, speaker_embeds)}

            yield similarity_dict


if __name__ == "__main__":
    # Load audio file
    # Source for the interview: https://www.youtube.com/watch?v=X2zqiX6yL3I
    audio_file = "data/youtube_test.wav"
    sd = SpeakerDiarization()
    similarity_dict = sd.simple_diarize(audio_file=audio_file, reference_audio_file=audio_file, visualization=True)
    print(similarity_dict)
