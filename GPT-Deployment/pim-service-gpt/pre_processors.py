from typing import List

Vector = List[str]

# BUCKET_NAME = io.etherlabs.artifacts
bucket = os.getenv('BUCKET_NAME', 'io.etherlabs.gpt.artifacts')
# TOKENIZER = staging2/tokenizer/english.pickle
mind_path = os.getenv('TOKENIZER')
tokenizer_dl_path = os.path.join(os.sep, 'tmp', 'english.pickle')
s3.Bucket(bucket).download_file(tokenizer_path,tokenizer_dl_path)
sent_tokenizer = pickle.load(open(tokenizer_dl_path,"rb"))

def preprocessSegments(transcript: str) -> Vector:
    processed_transcript_list = []
    if len(transcript.split('.')) > 1:
        for sentence in splitText(transcript):
            if len(sentence.split(' ')) > 3:
                processed_transcript_list.append(sentence.strip())

    # processed_transcript = ' '.join(processed_transcript_list)
    return processed_transcript_list

def splitText(text: str) -> Vector:
    # returns list of sentences
    if len(text)==0:
        return []
    text = text.strip()
    if not text.endswith((".","?","!")):
        text+="."
    text = text.replace("?.","?")
    split_text = sent_tokenizer.tokenize(text)
    return split_text
