from typing import List

Vector = List[str]


def preprocessSegments(transcript: str) -> Vector:
    processed_transcript_list = []
    if len(transcript.split('.')) > 1:
        for sentence in transcript.split('.'):
            if len(sentence.split(' ')) > 3:
                processed_transcript_list.append(sentence.strip())

    processed_transcript = '. '.join(processed_transcript_list)
    return processed_transcript
