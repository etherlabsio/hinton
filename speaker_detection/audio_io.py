import soundfile as sf
import librosa
import io
from pathlib import Path

from params import *


def read_wav(audio_file, source_sr=None):
    """

    Args:
        audio_file: Either location to audio file or str having file name
        sr: if passing an audio waveform, the sampling rate of the waveform before
                    preprocessing. After preprocessing, the waveform'speaker sampling rate will match the data
                    hyperparameters. If passing a filepath, the sampling rate will be automatically detected and
                    this argument will be ignored.

    Returns:

    """
    # Load the wav from disk if needed
    if isinstance(audio_file, str) or isinstance(audio_file, Path):
        wav, source_sr = librosa.load(audio_file, sr=None)
    else:
        wav = audio_file

    # Resample the wav
    if source_sr is not None:
        wav = librosa.resample(wav, source_sr, sampling_rate)

    return wav


def read_wav_as_blocks(audio_file: str, block_length: int=128, source_sr=None):
    if source_sr is None:
        source_sr = librosa.get_samplerate(audio_file)

    # Set the frame parameters to be equivalent to the librosa defaults
    # in the file's native sampling rate
    frame_length = (2048 * source_sr) // 22050
    hop_length = (512 * source_sr) // 22050

    # Stream the data, working on 128 frames at a time
    stream = librosa.stream(audio_file,
                            block_length=block_length,
                            frame_length=frame_length,
                            hop_length=hop_length)

    for fragments in stream:
        yield fragments


def read_wav_from_bytes(audio_file):

    with open(audio_file, 'rb') as f_:
        tmp = io.BytesIO(f_.read())
        wav_data, samplerate = sf.read(tmp)

        yield wav_data

def convert_to_wav(audio_file, output_file_name):
    sf.write(output_file_name, audio_file, samplerate=sampling_rate, subtype='PCM_24')

    return output_file_name