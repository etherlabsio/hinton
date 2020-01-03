import sys
sys.path.append("/home/ray__/CS/org/etherlabs/ai-engine/pkg/")

import text_preprocessing.preprocess as tp
from extra_preprocess import preprocess_text


def decode_meeting_file(request):
    request = request["body"]
    request["segments"] = sorted(request['segments'], key=lambda kv:kv['startTime'])
    for index, seg in enumerate(request["segments"]):
        request["segments"][index]["originalText"] = " ".join(preprocess_text(seg["originalText"]))
    segments_map = {}
    for index, seg in enumerate(request["segments"]):
        if seg["originalText"] != "":
            segments_map[seg['id']] = seg
            # if len(seg["originalText"].split(". "))==1 and len(seg["originalText"].split(" "))<=6 :
            #continue
            segments_map[seg['id']]["order"] = index
    text = list(map(lambda seg: (seg["originalText"], seg["id"]), [segment for segment in request['segments'] if segment["originalText"]!=""]))
    seg_list = [sent for sent, id  in text]
    segid_list = [id for sent, id in text]
    sent_list = list(map(lambda seg, segid:([sent + ". " for sent in seg.split(". ")],segid), seg_list, segid_list))
    sent_list = [(sent, segid) for seg, segid in sent_list for sent in seg]